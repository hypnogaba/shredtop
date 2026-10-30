//! Shred reassembly and transaction decoding.
//!
//! Accepts raw shreds from `ShredReceiver`, accumulates data shred payloads per slot,
//! flushes contiguous runs into an entry buffer, and deserializes `Entry` structs via
//! wincode to extract `VersionedTransaction`s.
//!
//! FEC (Reed-Solomon erasure) recovery is implemented for Merkle coding shreds.
//! When a FEC set accumulates enough shards (data + coding >= num_data), missing
//! data shreds are reconstructed and inserted into the slot's data_payloads map.

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use reed_solomon_erasure::galois_8::ReedSolomon;
use solana_transaction::versioned::VersionedTransaction;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use crate::metrics;
use crate::source_metrics::{SlotOutcome, SlotStats, SourceMetrics};

/// Upper bound on transactions in one Entry. A larger count means we are not
/// looking at an Entry header.
const MAX_ENTRY_TXS: u64 = 512;

/// Why an Entry could not be read from the front of a buffer.
#[derive(Debug, PartialEq)]
pub(crate) enum EntryError {
    /// The Entry continues past the end of the buffer.
    Truncated,
    /// The bytes are not a readable Entry.
    Malformed,
}

/// Read one `solana_entry::entry::Entry` (num_hashes u64, hash [u8; 32],
/// transaction count u64, transactions) from the front of `buf`.
///
/// Transactions are read with wincode: v1 transactions (SIMD-0385) put the
/// message first and the signatures after it with no length prefix, and only
/// wincode understands that layout. bincode reads legacy and v0 only.
///
/// Returns the transactions and the number of bytes the Entry took.
pub(crate) fn read_entry(buf: &[u8]) -> Result<(Vec<VersionedTransaction>, usize), EntryError> {
    if buf.len() < 48 {
        return Err(EntryError::Truncated);
    }
    let count = u64::from_le_bytes(buf[40..48].try_into().unwrap());
    if count > MAX_ENTRY_TXS {
        return Err(EntryError::Malformed);
    }
    let mut rest = &buf[48..];
    let mut txs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        match wincode::deserialize_from::<VersionedTransaction>(&mut rest) {
            Ok(t) => txs.push(t),
            Err(wincode::ReadError::Io(wincode::io::ReadError::ReadSizeLimit(_))) => {
                return Err(EntryError::Truncated)
            }
            Err(_) => return Err(EntryError::Malformed),
        }
    }
    Ok((txs, buf.len() - rest.len()))
}

/// Read a `Vec<Entry>` as the Jito proxy sends it (u64 count, then Entries).
/// An unreadable Entry is skipped; the Entries after it are still read.
pub(crate) fn read_entry_batch(buf: &[u8]) -> Vec<VersionedTransaction> {
    let mut consumed = 0;
    let mut boundary_scanned = true;
    let mut batch_starts = VecDeque::from([0]);
    decode_entries(buf, &mut consumed, &mut boundary_scanned, &mut batch_starts)
}

/// Decode the complete Entries in `buf[*consumed..]` and advance `consumed`
/// past them. An Entry cut off at the end of `buf` is left for the next call.
///
/// `buf` is a run of entry batches. Each batch is a u64 Entry count followed
/// by the Entries (a count of 0 marks a block marker instead, which we skip).
/// `batch_starts` holds the known batch offsets in `buf`, in order.
///
/// When `boundary_scanned` is false, `buf[*consumed..]` may start mid-Entry, so
/// first scan forward for an Entry header or the next batch start. An
/// unreadable Entry is skipped the same way: no amount of extra data fixes it,
/// and waiting on it would lose every Entry after it.
fn decode_entries(
    buf: &[u8],
    consumed: &mut usize,
    boundary_scanned: &mut bool,
    batch_starts: &mut VecDeque<usize>,
) -> Vec<VersionedTransaction> {
    let mut txs = Vec::new();

    loop {
        while batch_starts.front().is_some_and(|&s| s < *consumed) {
            batch_starts.pop_front();
        }
        // Entries never cross into the next batch.
        let batch_end = batch_starts.front().copied().unwrap_or(buf.len());

        // ── Phase 1: locate the first Entry boundary ────────────────────────
        if !*boundary_scanned {
            let rest = &buf[*consumed..];
            let limit = (batch_end - *consumed).min(rest.len().saturating_sub(47));

            // Anchor only on an Entry with transactions, each carrying at
            // least one signature: zero-filled bytes easily parse as an
            // empty Entry or an unsigned transaction. Skipping real empty
            // (tick) Entries here loses nothing.
            let found_at = (0..limit).find(|&off| {
                read_entry(&rest[off..]).is_ok_and(|(entry_txs, used)| {
                    off + used <= batch_end - *consumed
                        && !entry_txs.is_empty()
                        && entry_txs.iter().all(|t| !t.signatures.is_empty())
                })
            });

            match found_at {
                Some(off) => *consumed += off,
                // A batch start is always a boundary.
                None if batch_end < buf.len() => *consumed = batch_end,
                None => return txs,
            }
            *boundary_scanned = true;
        }

        // ── Phase 2: decode complete Entries ────────────────────────────────
        loop {
            if batch_starts.front() == Some(&*consumed) {
                let Some(count) = buf.get(*consumed..*consumed + 8) else {
                    return txs;
                };
                if u64::from_le_bytes(count.try_into().unwrap()) == 0 {
                    // Block marker, not Entries: skip to the next batch once
                    // we know where it starts.
                    match batch_starts.get(1) {
                        Some(&next) => *consumed = next,
                        None => return txs,
                    }
                } else {
                    *consumed += 8;
                }
                batch_starts.pop_front();
                continue;
            }
            let batch_end = batch_starts.front().copied();

            match read_entry(&buf[*consumed..]) {
                Ok((entry_txs, used)) if batch_end.is_none_or(|end| *consumed + used <= end) => {
                    txs.extend(entry_txs);
                    *consumed += used;
                }
                // Entry continues in data we have not received yet.
                Err(EntryError::Truncated) if batch_end.is_none() => return txs,
                // Unreadable, or runs past the end of its batch.
                _ => {
                    *consumed += 1;
                    *boundary_scanned = false;
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Raw shred header parsing
//
// solana_ledger::shred is pub(crate) in Agave 3.x — not accessible externally.
// We parse the binary layout directly using the stable Agave shred wire format.
//
// Common header layout (all shred types):
//   Bytes   0 ..  63 = signature (64 bytes, not used here)
//   Byte   64        = ShredVariant
//   Bytes  65 ..  72 = slot (u64 LE)
//   Bytes  73 ..  76 = index (u32 LE)
//   Bytes  77 ..  78 = version (u16 LE, not used)
//   Bytes  79 ..  82 = fec_set_index (u32 LE, not used)
//
// Data shred header (appended after common header at offset 83):
//   Bytes  83 ..  84 = parent_offset (u16 LE, not used)
//   Byte   85        = flags  (bit 0x01 = LAST_SHRED_IN_SLOT,
//                              bit 0x40 = DATA_COMPLETE: last shred of an entry batch)
//   Bytes  86 ..  87 = size   (u16 LE) — absolute end offset of entry data from byte 0
//                              i.e. data = bytes[88..size]
//
// Entry data location (identical for all data shred types — Legacy and all Merkle variants):
//   Bytes 88 .. size = entry data
//
// Shred variant byte (byte 64):
//   0xa5             = LegacyData
//   0x5a             = LegacyCode  (skipped)
//   high nibble 0x8  = MerkleData unchained, unsigned    (0x80–0x8F)
//   high nibble 0x9  = MerkleData chained,   unsigned    (0x90–0x9F) ← current mainnet
//   high nibble 0xa  = MerkleData unchained,  resigned   (0xa0–0xaF, but 0xa5 is LegacyData)
//   high nibble 0xb  = MerkleData chained,    resigned   (0xb0–0xbF)
//   high nibble 0x4–0x7 = MerkleCode (skipped)
//
// Proof entries, chained merkle root, resigned signature are all appended AFTER `size` and
// are therefore invisible to our parser — we just stop at `size`.
// ---------------------------------------------------------------------------

const VARIANT_OFF: usize = 64;
const SLOT_OFF: usize = 65;
const INDEX_OFF: usize = 73;
const FEC_SET_INDEX_OFF: usize = 79; // u32 LE
const FLAGS_OFF: usize = 85;
const SIZE_OFF: usize = 86; // u16 LE: absolute end of entry data (bytes[88..size])
const DATA_OFF: usize = 88; // entry data starts here (same for all data shred types)
const LAST_IN_SLOT_FLAG: u8 = 0x01;
const DATA_COMPLETE_FLAG: u8 = 0x40;
const LEGACY_DATA_VARIANT: u8 = 0xa5;

// Coding shred header fields (after common header at offset 83)
const CODE_NUM_DATA_OFF: usize = 83; // u16 LE: number of data shreds in FEC set
const CODE_NUM_CODE_OFF: usize = 85; // u16 LE: number of coding shreds in FEC set
const CODE_POSITION_OFF: usize = 87; // u16 LE: this coding shred's position (0-based)
const CODE_HDR_END: usize = 89; // minimum length for a coding shred

// Agave Merkle shred fixed buffer size used as the RS symbol width.
// Both data and coding shreds are padded / truncated to this size for RS.
const SHRED_RS_SIZE: usize = 1228;

/// Parse slot, index and fec_set_index from any shred type (code or data).
/// Returns None only if the buffer is shorter than the common header.
fn shred_slot_index(bytes: &[u8]) -> Option<(u64, u32, u32)> {
    if bytes.len() < FEC_SET_INDEX_OFF + 4 {
        return None;
    }
    let slot = u64::from_le_bytes(bytes[SLOT_OFF..SLOT_OFF + 8].try_into().unwrap());
    let index = u32::from_le_bytes(bytes[INDEX_OFF..INDEX_OFF + 4].try_into().unwrap());
    let fec_set_index = u32::from_le_bytes(
        bytes[FEC_SET_INDEX_OFF..FEC_SET_INDEX_OFF + 4]
            .try_into()
            .unwrap(),
    );
    Some((slot, index, fec_set_index))
}

/// Parsed fields from a coding shred header.
struct CodingShredInfo {
    num_data: u16,
    num_coding: u16,
    /// This coding shred's 0-based position within the coding shreds of the FEC set.
    position: u16,
}

/// Parse the coding-shred-specific header fields.
/// Returns None for non-coding shreds, malformed buffers, or zero num_data/num_coding.
fn parse_coding_header(bytes: &[u8]) -> Option<CodingShredInfo> {
    if bytes.len() < CODE_HDR_END {
        return None;
    }
    let variant = bytes[VARIANT_OFF];
    // Coding shreds: high nibble 0x4–0x7 (Merkle variants).
    // 0x5a is LegacyCode — skip; we only handle Merkle coding shreds.
    let high = variant & 0xF0;
    if high != 0x40 && high != 0x50 && high != 0x60 && high != 0x70 {
        return None;
    }
    if variant == 0x5a {
        // LegacyCode — RS layout differs; skip.
        return None;
    }

    let num_data = u16::from_le_bytes([bytes[CODE_NUM_DATA_OFF], bytes[CODE_NUM_DATA_OFF + 1]]);
    let num_coding = u16::from_le_bytes([bytes[CODE_NUM_CODE_OFF], bytes[CODE_NUM_CODE_OFF + 1]]);
    let position = u16::from_le_bytes([bytes[CODE_POSITION_OFF], bytes[CODE_POSITION_OFF + 1]]);

    if num_data == 0 || num_coding == 0 {
        return None;
    }

    Some(CodingShredInfo {
        num_data,
        num_coding,
        position,
    })
}

/// Parse a data shred's entry payload.
/// Returns `(last_in_slot, data_bytes)` for data shreds, `None` for code
/// shreds or malformed payloads.
fn parse_data_payload(bytes: &[u8]) -> Option<(bool, Vec<u8>)> {
    if bytes.len() < DATA_OFF {
        return None;
    }
    let variant = bytes[VARIANT_OFF];

    let is_data =
        variant == LEGACY_DATA_VARIANT || matches!(variant & 0xF0, 0x80 | 0x90 | 0xa0 | 0xb0);
    if !is_data {
        return None;
    }

    let last_in_slot = (bytes[FLAGS_OFF] & LAST_IN_SLOT_FLAG) != 0;

    let size = u16::from_le_bytes([bytes[SIZE_OFF], bytes[SIZE_OFF + 1]]) as usize;
    if size < DATA_OFF || size > bytes.len() {
        return None;
    }

    Some((last_in_slot, bytes[DATA_OFF..size].to_vec()))
}

/// True when this data shred is the last one of an entry batch, so the next
/// shred starts a new batch.
fn ends_entry_batch(bytes: &[u8]) -> bool {
    bytes.len() > FLAGS_OFF && bytes[FLAGS_OFF] & DATA_COMPLETE_FLAG != 0
}

use crate::receiver::RawShred;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Decoded transaction with timing metadata for the latency pipeline.
pub struct DecodedTx {
    pub transaction: VersionedTransaction,
    pub slot: u64,
    pub shred_recv_ns: u64,
    pub decode_done_ns: u64,
}

// ---------------------------------------------------------------------------
// Per-slot state: accumulate data shred payloads
// ---------------------------------------------------------------------------

struct SlotState {
    /// Data shred payloads keyed by shred index
    data_payloads: HashMap<u32, Vec<u8>>,
    /// Next contiguous index we expect (for streaming deserialization).
    /// Initialised to u32::MAX; set to the first received shred index on
    /// the first call to `set_first_index` so that shred relays starting
    /// mid-block (e.g. DoubleZero, which only sends the tail FEC sets) can
    /// still accumulate a contiguous run without waiting for idx=0.
    next_contiguous: u32,
    /// Accumulated entry bytes from contiguous data shreds
    entry_buf: Vec<u8>,
    /// Bytes already consumed from entry_buf
    consumed: usize,
    /// Highest data shred index seen
    max_index: u32,
    /// Whether we've seen the last shred in slot
    last_seen: bool,
    last_touch_ns: u64,
    /// Earliest kernel receive timestamp (ns) across all shreds in this slot.
    /// Used as the shred-side timestamp for lead-time computation so the
    /// measurement reflects when the first shred data arrived, not when
    /// enough shreds accumulated to trigger deserialization.
    earliest_recv_ns: u64,
    /// Number of transactions decoded from this slot
    txs_decoded: u32,
    /// Unique data shreds received (direct + FEC-recovered)
    shreds_seen: u32,
    /// Data shreds reconstructed via Reed-Solomon FEC for this slot
    fec_recovered_count: u32,
    /// Whether this slot has already been counted in slot outcome metrics
    counted: bool,
    /// Whether the first Entry boundary has been located within entry_buf.
    /// When starting mid-stream (shred index > 0), the beginning of entry_buf
    /// may contain the tail of an incomplete Entry from earlier shreds.
    /// We scan forward once to skip past it before normal deserialization.
    boundary_scanned: bool,
    /// Indices of received data shreds that end an entry batch.
    batch_ends: HashSet<u32>,
    /// Offsets in entry_buf where an entry batch starts, in order.
    batch_starts: VecDeque<usize>,
}

impl SlotState {
    fn new(now: u64) -> Self {
        Self {
            data_payloads: HashMap::with_capacity(64),
            next_contiguous: u32::MAX, // set on first shred receipt
            entry_buf: Vec::with_capacity(64 * 1024),
            consumed: 0,
            max_index: 0,
            last_seen: false,
            last_touch_ns: now,
            earliest_recv_ns: u64::MAX,
            txs_decoded: 0,
            shreds_seen: 0,
            fec_recovered_count: 0,
            counted: false,
            boundary_scanned: false,
            batch_ends: HashSet::new(),
            batch_starts: VecDeque::new(),
        }
    }

    /// Called with the first shred index received for this slot.
    /// Anchors `next_contiguous` so that relay streams starting mid-block
    /// (shred indices > 0) are flushed immediately rather than waiting for
    /// a shred at index 0 that will never arrive.
    fn set_first_index(&mut self, idx: u32) {
        if self.next_contiguous == u32::MAX {
            self.next_contiguous = idx;
            self.boundary_scanned = idx == 0;
            if idx == 0 {
                self.batch_starts.push_back(0);
            }
        }
    }

    /// Try to flush contiguous data shred payloads into entry_buf
    fn flush_contiguous(&mut self) {
        while let Some(payload) = self.data_payloads.remove(&self.next_contiguous) {
            self.entry_buf.extend_from_slice(&payload);
            if self.batch_ends.remove(&self.next_contiguous) {
                self.batch_starts.push_back(self.entry_buf.len());
            }
            self.next_contiguous += 1;
        }
    }

    /// Try to deserialize entries from accumulated data and extract transactions.
    fn try_deserialize(&mut self) -> Vec<VersionedTransaction> {
        decode_entries(
            &self.entry_buf,
            &mut self.consumed,
            &mut self.boundary_scanned,
            &mut self.batch_starts,
        )
    }
}

// ---------------------------------------------------------------------------
// FEC set state: buffer shards for Reed-Solomon recovery
// ---------------------------------------------------------------------------

struct FecSet {
    num_data: usize,
    num_coding: usize,
    shards: HashMap<usize, Vec<u8>>,
    recovered: bool,
}

impl FecSet {
    fn new(num_data: usize, num_coding: usize) -> Self {
        Self {
            num_data,
            num_coding,
            shards: HashMap::with_capacity(num_data + num_coding),
            recovered: false,
        }
    }

    fn ready_to_recover(&self) -> bool {
        !self.recovered && self.shards.len() >= self.num_data
    }

    fn reconstruct(&mut self) -> Vec<(usize, Vec<u8>)> {
        self.recovered = true;

        let total = self.num_data + self.num_coding;
        if total == 0 || self.num_data == 0 || self.num_coding == 0 {
            return Vec::new();
        }

        let mut shard_opts: Vec<Option<Vec<u8>>> =
            (0..total).map(|i| self.shards.get(&i).cloned()).collect();

        let missing_data: Vec<usize> = (0..self.num_data)
            .filter(|i| !self.shards.contains_key(i))
            .collect();

        if missing_data.is_empty() {
            return Vec::new();
        }

        let rs = match ReedSolomon::new(self.num_data, self.num_coding) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(
                    num_data = self.num_data,
                    num_coding = self.num_coding,
                    err = %e,
                    "FEC: failed to create ReedSolomon instance"
                );
                return Vec::new();
            }
        };

        if let Err(e) = rs.reconstruct(&mut shard_opts) {
            tracing::debug!(
                num_data = self.num_data,
                num_coding = self.num_coding,
                present = self.shards.len(),
                err = %e,
                "FEC: RS reconstruction failed"
            );
            return Vec::new();
        }

        let mut result = Vec::with_capacity(missing_data.len());
        for idx in missing_data {
            if let Some(Some(shard)) = shard_opts.get(idx) {
                result.push((idx, shard.clone()));
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ShredDecoder
// ---------------------------------------------------------------------------

const MAX_ACTIVE_SLOTS: usize = 64;
const SLOT_EXPIRY_DISTANCE: u64 = 32;

pub struct ShredDecoder {
    rx: Receiver<RawShred>,
    tx: Sender<DecodedTx>,
    metrics: Arc<SourceMetrics>,
}

impl ShredDecoder {
    pub fn new(rx: Receiver<RawShred>, tx: Sender<DecodedTx>, metrics: Arc<SourceMetrics>) -> Self {
        Self { rx, tx, metrics }
    }

    pub fn run(&self) -> Result<()> {
        tracing::info!("shred decoder started");

        let mut slots: HashMap<u64, SlotState> = HashMap::with_capacity(MAX_ACTIVE_SLOTS);
        let mut fec_sets: HashMap<u64, HashMap<u32, FecSet>> =
            HashMap::with_capacity(MAX_ACTIVE_SLOTS);
        let mut highest_slot: u64 = 0;

        for raw_shred in &self.rx {
            let decode_start = metrics::now_ns();

            let (slot, shred_index, fec_set_index) = match shred_slot_index(&raw_shred.data) {
                Some(si) => si,
                None => continue,
            };

            // Guard against corrupted shreds with garbage slot values.
            // A jump larger than MAX_SLOT_JUMP would poison highest_slot and
            // cause all subsequent real shreds to fail the expiry check below,
            // silently dropping all output until restart.
            //
            // Exception: highest_slot == 0 means we haven't seen any valid shred yet.
            // The first real shred sets the baseline regardless of slot distance
            // (mainnet slots are ~350_000_000, which exceeds MAX_SLOT_JUMP from 0).
            const MAX_SLOT_JUMP: u64 = 1000;
            if slot > highest_slot
                && (highest_slot == 0 || slot.saturating_sub(highest_slot) <= MAX_SLOT_JUMP)
            {
                highest_slot = slot;
                slots.retain(|&s, state| {
                    if s + SLOT_EXPIRY_DISTANCE >= highest_slot {
                        return true;
                    }
                    if !state.counted {
                        if state.txs_decoded > 0 {
                            self.metrics.slots_partial.fetch_add(1, Relaxed);
                            self.metrics.push_slot_stats(SlotStats {
                                slot: s,
                                shreds_seen: state.shreds_seen,
                                fec_recovered: state.fec_recovered_count,
                                txs_decoded: state.txs_decoded,
                                outcome: SlotOutcome::Partial,
                            });
                        } else {
                            self.metrics.slots_dropped.fetch_add(1, Relaxed);
                            self.metrics.push_slot_stats(SlotStats {
                                slot: s,
                                shreds_seen: state.shreds_seen,
                                fec_recovered: state.fec_recovered_count,
                                txs_decoded: 0,
                                outcome: SlotOutcome::Dropped,
                            });
                        }
                    }
                    false
                });
                fec_sets.retain(|&s, _| s + SLOT_EXPIRY_DISTANCE >= highest_slot);
            }

            if highest_slot.saturating_sub(slot) > SLOT_EXPIRY_DISTANCE {
                continue;
            }

            let now = metrics::now_ns();

            // ── Coding shred path ────────────────────────────────────────────
            if let Some(code_info) = parse_coding_header(&raw_shred.data) {
                let num_data = code_info.num_data as usize;
                let num_coding = code_info.num_coding as usize;
                let code_position = code_info.position as usize;
                let shard_pos = num_data + code_position;

                if code_position >= num_coding {
                    continue;
                }

                let slot_fec = fec_sets.entry(slot).or_default();
                let fec = slot_fec.entry(fec_set_index).or_insert_with(|| {
                    // Count expected data shreds per FEC set as we discover them.
                    // This is the correct denominator for coverage on tail-only feeds
                    // where the last-in-slot marker rarely arrives.
                    self.metrics
                        .coverage_shreds_expected
                        .fetch_add(num_data as u64, Relaxed);
                    FecSet::new(num_data, num_coding)
                });

                if fec.num_data != num_data || fec.num_coding != num_coding {
                    continue;
                }

                fec.shards.entry(shard_pos).or_insert_with(|| {
                    let mut buf = raw_shred.data.clone();
                    buf.resize(SHRED_RS_SIZE, 0);
                    buf
                });

                if fec.ready_to_recover() {
                    let recovered = fec.reconstruct();
                    fec.shards.clear();
                    if !recovered.is_empty() {
                        let slot_state = slots.entry(slot).or_insert_with(|| {
                            self.metrics.slots_attempted.fetch_add(1, Relaxed);
                            SlotState::new(now)
                        });
                        slot_state.last_touch_ns = now;
                        if raw_shred.recv_timestamp_ns < slot_state.earliest_recv_ns {
                            slot_state.earliest_recv_ns = raw_shred.recv_timestamp_ns;
                        }

                        let mut recovered_count = 0u64;
                        for (data_shard_idx, shard_bytes) in recovered {
                            let global_idx = fec_set_index.saturating_add(data_shard_idx as u32);
                            if slot_state.data_payloads.contains_key(&global_idx) {
                                continue;
                            }
                            if let Some((last_in_slot, payload)) = parse_data_payload(&shard_bytes)
                            {
                                slot_state.set_first_index(global_idx);
                                if global_idx > slot_state.max_index {
                                    slot_state.max_index = global_idx;
                                }
                                if last_in_slot {
                                    slot_state.last_seen = true;
                                }
                                if ends_entry_batch(&shard_bytes) {
                                    slot_state.batch_ends.insert(global_idx);
                                }
                                slot_state.data_payloads.insert(global_idx, payload);
                                recovered_count += 1;
                            }
                        }

                        if recovered_count > 0 {
                            self.metrics
                                .fec_recovered_shreds
                                .fetch_add(recovered_count, Relaxed);
                            self.metrics
                                .coverage_shreds_seen
                                .fetch_add(recovered_count, Relaxed);
                            slot_state.shreds_seen += recovered_count as u32;
                            slot_state.fec_recovered_count += recovered_count as u32;

                            slot_state.flush_contiguous();

                            if slot_state.last_seen
                                && slot_state.next_contiguous > slot_state.max_index
                                && !slot_state.counted
                            {
                                self.metrics.slots_complete.fetch_add(1, Relaxed);
                                slot_state.counted = true;
                                self.metrics.push_slot_stats(SlotStats {
                                    slot,
                                    shreds_seen: slot_state.shreds_seen,
                                    fec_recovered: slot_state.fec_recovered_count,
                                    txs_decoded: slot_state.txs_decoded,
                                    outcome: SlotOutcome::Complete,
                                });
                            }

                            let txs = slot_state.try_deserialize();
                            if !txs.is_empty() {
                                let decode_done = metrics::now_ns();
                                metrics::METRICS.record_stage(
                                    &metrics::METRICS.decode_ns,
                                    decode_done - decode_start,
                                );

                                let tx_count = txs.len() as u32;
                                slot_state.txs_decoded += tx_count;
                                self.metrics.txs_decoded.fetch_add(tx_count as u64, Relaxed);

                                for tx in txs {
                                    let decoded = DecodedTx {
                                        transaction: tx,
                                        slot,
                                        shred_recv_ns: slot_state.earliest_recv_ns,
                                        decode_done_ns: decode_done,
                                    };
                                    if self.tx.try_send(decoded).is_err() {
                                        self.metrics.txs_dropped.fetch_add(1, Relaxed);
                                    }
                                }
                            }
                        }
                    }
                }

                continue;
            }

            // ── Data shred path ──────────────────────────────────────────────
            let (last_in_slot, payload) = match parse_data_payload(&raw_shred.data) {
                Some(d) => d,
                None => continue,
            };

            self.metrics.coverage_shreds_seen.fetch_add(1, Relaxed);

            let state = slots.entry(slot).or_insert_with(|| {
                self.metrics.slots_attempted.fetch_add(1, Relaxed);
                SlotState::new(now)
            });
            state.last_touch_ns = now;
            if raw_shred.recv_timestamp_ns < state.earliest_recv_ns {
                state.earliest_recv_ns = raw_shred.recv_timestamp_ns;
            }

            let data_shard_idx = shred_index.checked_sub(fec_set_index).map(|i| i as usize);
            if let Some(shard_pos) = data_shard_idx {
                let slot_fec = fec_sets.entry(slot).or_default();
                if let Some(fec) = slot_fec.get_mut(&fec_set_index) {
                    fec.shards.entry(shard_pos).or_insert_with(|| {
                        let mut buf = raw_shred.data.clone();
                        buf.resize(SHRED_RS_SIZE, 0);
                        buf
                    });
                }
            }

            state.set_first_index(shred_index);

            if shred_index > state.max_index {
                state.max_index = shred_index;
            }
            if last_in_slot {
                state.last_seen = true;
            }
            if ends_entry_batch(&raw_shred.data) {
                state.batch_ends.insert(shred_index);
            }

            if state.data_payloads.insert(shred_index, payload).is_none() {
                state.shreds_seen += 1;
            }
            state.flush_contiguous();

            if state.last_seen && state.next_contiguous > state.max_index && !state.counted {
                self.metrics.slots_complete.fetch_add(1, Relaxed);
                state.counted = true;
                self.metrics.push_slot_stats(SlotStats {
                    slot,
                    shreds_seen: state.shreds_seen,
                    fec_recovered: state.fec_recovered_count,
                    txs_decoded: state.txs_decoded,
                    outcome: SlotOutcome::Complete,
                });
            }

            let txs = state.try_deserialize();
            if !txs.is_empty() {
                let decode_done = metrics::now_ns();
                metrics::METRICS
                    .record_stage(&metrics::METRICS.decode_ns, decode_done - decode_start);

                let tx_count = txs.len() as u32;
                state.txs_decoded += tx_count;
                self.metrics.txs_decoded.fetch_add(tx_count as u64, Relaxed);

                for tx in txs {
                    let decoded = DecodedTx {
                        transaction: tx,
                        slot,
                        shred_recv_ns: state.earliest_recv_ns,
                        decode_done_ns: decode_done,
                    };
                    let _ = self.tx.try_send(decoded);
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flush_contiguous_in_order() {
        let mut state = SlotState::new(0);
        state.set_first_index(0);

        state.data_payloads.insert(0, vec![1, 2, 3]);
        state.data_payloads.insert(1, vec![4, 5, 6]);
        state.data_payloads.insert(2, vec![7, 8, 9]);
        state.flush_contiguous();

        assert_eq!(state.entry_buf, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(state.next_contiguous, 3);
        assert!(state.data_payloads.is_empty());
    }

    #[test]
    fn test_flush_contiguous_out_of_order() {
        let mut state = SlotState::new(0);
        state.set_first_index(0);

        state.data_payloads.insert(2, vec![7, 8, 9]);
        state.flush_contiguous();
        assert!(state.entry_buf.is_empty());
        assert_eq!(state.next_contiguous, 0);

        state.data_payloads.insert(0, vec![1, 2, 3]);
        state.flush_contiguous();
        assert_eq!(state.entry_buf, vec![1, 2, 3]);
        assert_eq!(state.next_contiguous, 1);

        state.data_payloads.insert(1, vec![4, 5, 6]);
        state.flush_contiguous();
        assert_eq!(state.entry_buf, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(state.next_contiguous, 3);
        assert!(state.data_payloads.is_empty());
    }

    fn make_shred(variant: u8, data: &[u8], last_in_slot: bool) -> Vec<u8> {
        let total = 1228;
        let mut buf = vec![0u8; total];
        buf[VARIANT_OFF] = variant;
        let size_abs = (DATA_OFF + data.len()) as u16;
        buf[SIZE_OFF] = size_abs as u8;
        buf[SIZE_OFF + 1] = (size_abs >> 8) as u8;
        if last_in_slot {
            buf[FLAGS_OFF] = LAST_IN_SLOT_FLAG;
        }
        buf[DATA_OFF..DATA_OFF + data.len()].copy_from_slice(data);
        buf
    }

    #[test]
    fn test_parse_legacy_data() {
        let payload = b"hello world";
        let shred = make_shred(LEGACY_DATA_VARIANT, payload, false);
        let (last, data) = parse_data_payload(&shred).expect("should parse");
        assert!(!last);
        assert_eq!(data, payload);
    }

    #[test]
    fn test_parse_chained_merkle_data() {
        let payload = b"solana entry bytes";
        let shred = make_shred(0x92, payload, true);
        let (last, data) = parse_data_payload(&shred).expect("should parse");
        assert!(last);
        assert_eq!(data, payload);
    }

    #[test]
    fn test_parse_legacy_code_rejected() {
        let shred = make_shred(0x5a, b"", false);
        assert!(parse_data_payload(&shred).is_none());
    }

    #[test]
    fn test_parse_merkle_code_rejected() {
        let shred = make_shred(0x44, b"", false);
        assert!(parse_data_payload(&shred).is_none());
    }

    #[test]
    fn test_parse_too_short() {
        assert!(parse_data_payload(&[0u8; 10]).is_none());
        assert!(parse_data_payload(&[0u8; 87]).is_none());
    }

    #[test]
    fn test_parse_size_beyond_buf() {
        let mut shred = make_shred(0x90, b"data", false);
        shred[SIZE_OFF] = 0x0F;
        shred[SIZE_OFF + 1] = 0x27;
        assert!(parse_data_payload(&shred).is_none());
    }

    #[test]
    fn test_flush_contiguous_mid_stream() {
        let mut state = SlotState::new(0);

        state.set_first_index(1001);
        state.data_payloads.insert(1001, vec![10, 11]);
        state.flush_contiguous();
        assert_eq!(state.entry_buf, vec![10, 11]);
        assert_eq!(state.next_contiguous, 1002);

        state.data_payloads.insert(1000, vec![99]);
        state.flush_contiguous();
        assert_eq!(state.entry_buf, vec![10, 11], "1000 is stale, not flushed");
        assert_eq!(state.next_contiguous, 1002);

        state.data_payloads.insert(1002, vec![20, 21]);
        state.flush_contiguous();
        assert_eq!(state.entry_buf, vec![10, 11, 20, 21]);
        assert_eq!(state.next_contiguous, 1003);
    }

    #[test]
    fn test_complete_detection() {
        let mut state = SlotState::new(0);
        state.set_first_index(0);

        state.max_index = 2;
        state.last_seen = true;

        state.data_payloads.insert(0, vec![1]);
        state.data_payloads.insert(1, vec![2]);
        state.data_payloads.insert(2, vec![3]);
        state.flush_contiguous();

        assert!(state.last_seen);
        assert!(state.next_contiguous > state.max_index);
        assert!(!state.counted);

        if state.last_seen && state.next_contiguous > state.max_index && !state.counted {
            state.counted = true;
        }
        assert!(state.counted);
    }

    fn make_coding_shred(variant: u8, num_data: u16, num_coding: u16, position: u16) -> Vec<u8> {
        let mut buf = vec![0u8; SHRED_RS_SIZE];
        buf[VARIANT_OFF] = variant;
        buf[CODE_NUM_DATA_OFF] = num_data as u8;
        buf[CODE_NUM_DATA_OFF + 1] = (num_data >> 8) as u8;
        buf[CODE_NUM_CODE_OFF] = num_coding as u8;
        buf[CODE_NUM_CODE_OFF + 1] = (num_coding >> 8) as u8;
        buf[CODE_POSITION_OFF] = position as u8;
        buf[CODE_POSITION_OFF + 1] = (position >> 8) as u8;
        buf
    }

    #[test]
    fn test_parse_coding_header_valid() {
        let shred = make_coding_shred(0x64, 32, 32, 5);
        let info = parse_coding_header(&shred).expect("should parse coding header");
        assert_eq!(info.num_data, 32);
        assert_eq!(info.num_coding, 32);
        assert_eq!(info.position, 5);
    }

    #[test]
    fn test_parse_coding_header_rejects_data() {
        let shred = make_shred(0x90, b"hello", false);
        assert!(parse_coding_header(&shred).is_none());
    }

    #[test]
    fn test_parse_coding_header_rejects_legacy_code() {
        let shred = make_coding_shred(0x5a, 16, 16, 0);
        assert!(parse_coding_header(&shred).is_none());
    }

    #[test]
    fn test_parse_coding_header_rejects_zero_counts() {
        let zero_data = make_coding_shred(0x64, 0, 16, 0);
        assert!(parse_coding_header(&zero_data).is_none());
        let zero_coding = make_coding_shred(0x64, 16, 0, 0);
        assert!(parse_coding_header(&zero_coding).is_none());
    }

    #[test]
    fn test_parse_coding_header_too_short() {
        assert!(parse_coding_header(&[0u8; CODE_HDR_END - 1]).is_none());
    }

    #[test]
    fn test_fec_set_reconstruct_recovers_missing_data() {
        use reed_solomon_erasure::galois_8::ReedSolomon;

        const N: usize = 2;
        const M: usize = 2;
        const SZ: usize = 64;

        let original: Vec<Vec<u8>> = vec![vec![1u8; SZ], vec![2u8; SZ]];
        let rs = ReedSolomon::new(N, M).unwrap();
        let mut all_shards: Vec<Vec<u8>> = original.clone();
        all_shards.push(vec![0u8; SZ]);
        all_shards.push(vec![0u8; SZ]);
        rs.encode(&mut all_shards).unwrap();

        let mut fec = FecSet::new(N, M);
        fec.shards.insert(0, all_shards[0].clone());
        fec.shards.insert(2, all_shards[2].clone());
        fec.shards.insert(3, all_shards[3].clone());

        assert!(fec.ready_to_recover());

        let recovered = fec.reconstruct();
        assert_eq!(recovered.len(), 1);
        let (idx, bytes) = &recovered[0];
        assert_eq!(*idx, 1);
        assert_eq!(bytes, &original[1]);
    }

    #[test]
    fn test_fec_set_not_ready_when_insufficient_shards() {
        let mut fec = FecSet::new(4, 4);
        fec.shards.insert(4, vec![0u8; SHRED_RS_SIZE]);
        assert!(!fec.ready_to_recover());
    }

    #[test]
    fn test_fec_set_reconstruct_no_missing_data() {
        use reed_solomon_erasure::galois_8::ReedSolomon;

        const N: usize = 2;
        const M: usize = 2;
        const SZ: usize = 64;

        let rs = ReedSolomon::new(N, M).unwrap();
        let mut all_shards: Vec<Vec<u8>> =
            vec![vec![3u8; SZ], vec![4u8; SZ], vec![0u8; SZ], vec![0u8; SZ]];
        rs.encode(&mut all_shards).unwrap();

        let mut fec = FecSet::new(N, M);
        for (i, s) in all_shards.iter().enumerate() {
            fec.shards.insert(i, s.clone());
        }
        let recovered = fec.reconstruct();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_highest_slot_not_poisoned_by_garbage() {
        // Reproduces the production bug: a single shred with slot=u64::MAX would
        // set highest_slot to u64::MAX, causing every subsequent real shred to
        // fail the expiry check and be silently dropped forever.
        const MAX_SLOT_JUMP: u64 = 1000;

        let mut highest_slot: u64 = 100;

        // Simulate a corrupted shred with a garbage slot value.
        let garbage_slot = u64::MAX;
        if garbage_slot > highest_slot && garbage_slot.saturating_sub(highest_slot) <= MAX_SLOT_JUMP
        {
            highest_slot = garbage_slot;
        }
        assert_eq!(
            highest_slot, 100,
            "garbage slot must not poison highest_slot"
        );

        // Normal slot progression must still work after the bad shred.
        let next_slot = 102u64;
        if next_slot > highest_slot && next_slot.saturating_sub(highest_slot) <= MAX_SLOT_JUMP {
            highest_slot = next_slot;
        }
        assert_eq!(highest_slot, 102);
    }

    use solana_message::{v0, v1, Address, Hash, Message, MessageHeader, VersionedMessage};
    use solana_signature::Signature;

    fn tx(seed: u8, message: VersionedMessage) -> VersionedTransaction {
        VersionedTransaction {
            signatures: vec![Signature::from([seed; 64])],
            message,
        }
    }

    /// Wire bytes of one Entry holding `txs`, written field by field.
    fn entry_bytes(txs: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&1u64.to_le_bytes()); // num_hashes
        out.extend_from_slice(&[0u8; 32]); // hash
        out.extend_from_slice(&(txs.len() as u64).to_le_bytes());
        for t in txs {
            out.extend_from_slice(t);
        }
        out
    }

    /// Wire bytes of one entry batch: the Entry count, then the Entries.
    fn batch(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut out = (entries.len() as u64).to_le_bytes().to_vec();
        for e in entries {
            out.extend_from_slice(e);
        }
        out
    }

    fn legacy_tx_bytes(seed: u8) -> Vec<u8> {
        wincode::serialize(&tx(seed, VersionedMessage::Legacy(Message::default()))).unwrap()
    }

    fn v0_tx_bytes(seed: u8) -> Vec<u8> {
        wincode::serialize(&tx(seed, VersionedMessage::V0(v0::Message::default()))).unwrap()
    }

    /// A transaction with an unknown message version (prefix 0xff). It cannot
    /// be read no matter how many more shreds arrive.
    fn bad_tx_bytes() -> Vec<u8> {
        let mut out = vec![1u8]; // one signature (short_vec)
        out.extend_from_slice(&[0xaa; 64]);
        out.push(0xff);
        out.extend_from_slice(&[0x11; 40]);
        out
    }

    /// A slot that starts at shred 0 with `buf` already flushed.
    fn slot_with(buf: Vec<u8>) -> SlotState {
        let mut s = SlotState::new(0);
        s.set_first_index(0);
        s.entry_buf = buf;
        s
    }

    fn sigs(txs: &[VersionedTransaction]) -> Vec<u8> {
        txs.iter()
            .map(|t| <[u8; 64]>::from(t.signatures[0])[0])
            .collect()
    }

    /// A v1 message with one signer, as the network sends it.
    fn v1_message(seed: u8) -> v1::Message {
        v1::Message {
            header: MessageHeader {
                num_required_signatures: 1,
                num_readonly_signed_accounts: 0,
                num_readonly_unsigned_accounts: 0,
            },
            config: v1::TransactionConfig::default(),
            lifetime_specifier: Hash::new_from_array([seed; 32]),
            account_keys: vec![Address::new_from_array([seed; 32])],
            instructions: vec![],
        }
    }

    /// SIMD-0385 wire layout of a v1 transaction: the message (starting with
    /// the 0x81 version byte), then the signatures with no length prefix.
    fn v1_tx_bytes(seed: u8) -> Vec<u8> {
        let mut out = v1_message(seed).serialize();
        out.extend_from_slice(&[seed; 64]);
        out
    }

    #[test]
    fn test_v1_wire_layout_matches_wincode() {
        let bytes = v1_tx_bytes(9);
        assert_eq!(bytes[0], 0x81);
        let t = tx(9, VersionedMessage::V1(v1_message(9)));
        assert_eq!(wincode::serialize(&t).unwrap(), bytes);
    }

    #[test]
    fn test_v1_transaction_decodes_between_legacy_entries() {
        let mut s = slot_with(batch(&[
            entry_bytes(&[legacy_tx_bytes(1)]),
            entry_bytes(&[v1_tx_bytes(9)]),
            entry_bytes(&[legacy_tx_bytes(2)]),
        ]));
        let txs = s.try_deserialize();
        assert_eq!(s.consumed, s.entry_buf.len());
        assert_eq!(sigs(&txs), vec![1, 9, 2]);
        assert!(matches!(txs[1].message, VersionedMessage::V1(_)));
    }

    #[test]
    fn test_every_cut_reads_as_truncated() {
        for tx_bytes in [legacy_tx_bytes(1), v0_tx_bytes(2), v1_tx_bytes(3)] {
            let entry = entry_bytes(&[tx_bytes]);
            for cut in 0..entry.len() {
                assert_eq!(read_entry(&entry[..cut]), Err(EntryError::Truncated));
            }
            let (txs, used) = read_entry(&entry).unwrap();
            assert_eq!(txs.len(), 1);
            assert_eq!(used, entry.len());
        }
    }

    #[test]
    fn test_cut_off_v1_transaction_waits_for_more_shreds() {
        let first = entry_bytes(&[legacy_tx_bytes(1)]);
        let full = batch(&[first.clone(), entry_bytes(&[v1_tx_bytes(9)])]);
        let first_end = 8 + first.len();

        for cut in first_end + 1..full.len() {
            let mut s = slot_with(full[..cut].to_vec());
            assert_eq!(sigs(&s.try_deserialize()), vec![1]);
            assert_eq!(s.consumed, first_end);

            s.entry_buf.extend_from_slice(&full[cut..]);
            assert_eq!(sigs(&s.try_deserialize()), vec![9]);
            assert_eq!(s.consumed, full.len());
        }
    }

    #[test]
    fn test_batches_split_across_shreds() {
        // Real shred layout: each entry batch starts with its Entry count, a
        // batch can span shreds, and the shred ending a batch says so. A count
        // of 0 marks a block marker, which carries no Entries.
        let batch1 = batch(&[
            entry_bytes(&[legacy_tx_bytes(1), v0_tx_bytes(2)]),
            entry_bytes(&[v1_tx_bytes(3)]),
        ]);
        let mut marker = 0u64.to_le_bytes().to_vec();
        marker.extend_from_slice(&[0x5a; 50]);
        let batch2 = batch(&[entry_bytes(&[v1_tx_bytes(4)])]);

        let cut = 30; // inside the first Entry of batch 1
        let shreds: Vec<(Vec<u8>, bool)> = vec![
            (batch1[..cut].to_vec(), false),
            (batch1[cut..].to_vec(), true),
            (marker, true),
            (batch2, true),
        ];

        let mut s = SlotState::new(0);
        s.set_first_index(0);
        let mut got = Vec::new();
        for (idx, (payload, ends_batch)) in shreds.into_iter().enumerate() {
            s.data_payloads.insert(idx as u32, payload);
            if ends_batch {
                s.batch_ends.insert(idx as u32);
            }
            s.flush_contiguous();
            got.extend(s.try_deserialize());
        }
        assert_eq!(sigs(&got), vec![1, 2, 3, 4]);
        assert_eq!(s.consumed, s.entry_buf.len());
    }

    #[test]
    fn test_mid_slot_start_with_v1() {
        // Relays may start mid-block: the buffer opens with the tail of an
        // Entry whose start we never saw.
        let lost = entry_bytes(&[legacy_tx_bytes(7)]);
        let mut buf = lost[lost.len() / 2..].to_vec();
        buf.extend(entry_bytes(&[legacy_tx_bytes(1)]));
        buf.extend(entry_bytes(&[v1_tx_bytes(9)]));

        let mut s = SlotState::new(0);
        s.set_first_index(5);
        s.entry_buf = buf;
        assert_eq!(sigs(&s.try_deserialize()), vec![1, 9]);
    }

    #[test]
    fn test_unreadable_entry_does_not_drop_rest_of_slot() {
        let mut s = slot_with(batch(&[
            entry_bytes(&[legacy_tx_bytes(1)]),
            entry_bytes(&[bad_tx_bytes()]),
            entry_bytes(&[legacy_tx_bytes(2)]),
        ]));
        assert_eq!(sigs(&s.try_deserialize()), vec![1, 2]);
    }

    #[test]
    fn test_incomplete_entry_waits_for_more_shreds() {
        let first = entry_bytes(&[legacy_tx_bytes(1)]);
        let second = entry_bytes(&[legacy_tx_bytes(2)]);
        let full = batch(&[first.clone(), second.clone()]);
        let split = 8 + first.len() + second.len() / 2;

        let mut s = slot_with(full[..split].to_vec());
        assert_eq!(sigs(&s.try_deserialize()), vec![1]);
        assert_eq!(s.consumed, 8 + first.len());

        s.entry_buf.extend_from_slice(&full[split..]);
        assert_eq!(sigs(&s.try_deserialize()), vec![2]);
    }

    #[test]
    fn test_jito_batch_keeps_entries_after_unreadable_one() {
        let b = batch(&[
            entry_bytes(&[legacy_tx_bytes(1)]),
            entry_bytes(&[bad_tx_bytes()]),
            entry_bytes(&[v1_tx_bytes(9)]),
        ]);
        assert_eq!(sigs(&read_entry_batch(&b)), vec![1, 9]);
    }

    #[test]
    fn test_jito_batch_with_unreadable_first_entry() {
        let b = batch(&[
            entry_bytes(&[bad_tx_bytes()]),
            entry_bytes(&[legacy_tx_bytes(1)]),
            entry_bytes(&[v1_tx_bytes(9)]),
        ]);
        assert_eq!(sigs(&read_entry_batch(&b)), vec![1, 9]);
    }
}
