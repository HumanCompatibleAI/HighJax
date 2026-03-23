//! Deduplication layer for tracing: suppresses repeated log messages.
//!
//! First occurrence of each unique message passes through. Subsequent identical
//! messages are suppressed. When a message has been suppressed N times
//! (N = 10, 100, 1000, ...) a summary line is emitted.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

/// A tracing layer that deduplicates repeated log messages.
pub struct DedupLayer {
    seen: Mutex<HashMap<u64, u64>>, // hash → suppression count
}

impl DedupLayer {
    pub fn new() -> Self {
        Self {
            seen: Mutex::new(HashMap::new()),
        }
    }
}

impl<S> tracing_subscriber::Layer<S> for DedupLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // We only filter — actual formatting is done by other layers.
        // This method is a no-op; filtering happens in `enabled` + `event_enabled`.
        let _ = event;
    }

    fn event_enabled(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        let hash = event_hash(event);

        // Extract count and summary info under the lock, then drop it before
        // emitting any tracing events (to avoid deadlock from re-entrancy).
        let (count, summary) = {
            let mut seen = match self.seen.lock() {
                Ok(g) => g,
                Err(_) => return true, // Don't suppress if lock is poisoned
            };

            let count = seen.entry(hash).or_insert(0);
            *count += 1;

            if *count == 1 {
                return true;
            }

            let summary = if count.is_power_of_ten() {
                Some((*count, EventMessage::extract(event)))
            } else {
                None
            };

            (*count, summary)
        }; // mutex dropped here

        let _ = count;
        if let Some((n, msg)) = summary {
            tracing::warn!("(repeated {} times, suppressing further) {}", n, msg);
        }

        false
    }
}

trait PowerOfTen {
    fn is_power_of_ten(&self) -> bool;
}

impl PowerOfTen for u64 {
    fn is_power_of_ten(&self) -> bool {
        if *self == 0 {
            return false;
        }
        let mut n = *self;
        while n % 10 == 0 {
            n /= 10;
        }
        n == 1
    }
}

fn event_hash(event: &tracing::Event<'_>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Hash the callsite (file + line) for uniqueness
    event.metadata().callsite().hash(&mut hasher);
    // Also hash the formatted message
    let msg = EventMessage::extract(event);
    msg.hash(&mut hasher);
    hasher.finish()
}

/// Visitor that extracts the formatted message from a tracing event.
struct EventMessage(String);

impl EventMessage {
    fn extract(event: &tracing::Event<'_>) -> String {
        let mut visitor = EventMessage(String::new());
        event.record(&mut visitor);
        visitor.0
    }
}

impl tracing::field::Visit for EventMessage {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{:?}", value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_of_ten() {
        assert!(!0u64.is_power_of_ten());
        assert!(1u64.is_power_of_ten());
        assert!(10u64.is_power_of_ten());
        assert!(100u64.is_power_of_ten());
        assert!(1000u64.is_power_of_ten());
        assert!(!2u64.is_power_of_ten());
        assert!(!15u64.is_power_of_ten());
        assert!(!99u64.is_power_of_ten());
    }
}
