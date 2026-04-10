//! `PtxWriter` — indent-aware string builder for PTX text emission.

use std::fmt::{self, Display};

/// An indent-aware string builder that produces formatted PTX text.
///
/// All [`Emit`](super::Emit) impls write through `PtxWriter`'s methods
/// rather than appending to the buffer directly. The
/// [`instruction`](Self::instruction) method is the central formatting
/// chokepoint — it handles indentation, operand comma-separation, and
/// the trailing semicolon.
///
/// # Sprint 1.5 note
///
/// `&[&dyn Display]` for operands creates some friction at call sites
/// (explicit `&` refs). Consider adding a `ptx_instr!()` macro for
/// ergonomics when this gets heavy use.
pub struct PtxWriter {
    buf: String,
    indent_level: usize,
    label_counter: u32,
}

const INDENT: &str = "    ";

impl Default for PtxWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl PtxWriter {
    /// Create a new writer with an empty buffer.
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            indent_level: 0,
            label_counter: 0,
        }
    }

    /// Write an indented line followed by a newline.
    pub fn line(&mut self, s: &str) -> fmt::Result {
        for _ in 0..self.indent_level {
            self.buf.push_str(INDENT);
        }
        self.buf.push_str(s);
        self.buf.push('\n');
        Ok(())
    }

    /// Write a blank line.
    pub fn blank(&mut self) -> fmt::Result {
        self.buf.push('\n');
        Ok(())
    }

    /// Increase indentation by one level.
    pub fn indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation by one level.
    pub fn dedent(&mut self) {
        self.indent_level = self.indent_level.saturating_sub(1);
    }

    /// Emit a PTX instruction with comma-separated operands and a trailing
    /// semicolon.
    ///
    /// Example output: `    add.f32 %f0, %f1, %f2;\n`
    pub fn instruction(&mut self, mnemonic: &str, operands: &[&dyn Display]) -> fmt::Result {
        for _ in 0..self.indent_level {
            self.buf.push_str(INDENT);
        }
        self.buf.push_str(mnemonic);
        if !operands.is_empty() {
            self.buf.push(' ');
            for (i, op) in operands.iter().enumerate() {
                if i > 0 {
                    self.buf.push_str(", ");
                }
                self.buf.push_str(&op.to_string());
            }
        }
        self.buf.push_str(";\n");
        Ok(())
    }

    /// Generate a unique label with the given prefix.
    ///
    /// Returns labels like `EXIT_0`, `LOOP_1`, `THEN_2`, etc.
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        let label = format!("{prefix}_{}", self.label_counter);
        self.label_counter += 1;
        label
    }

    /// Consume the writer and return the accumulated PTX text.
    pub fn finish(self) -> String {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_with_no_indent() {
        let mut w = PtxWriter::new();
        w.line(".version 7.8").unwrap();
        assert_eq!(w.finish(), ".version 7.8\n");
    }

    #[test]
    fn line_with_indent() {
        let mut w = PtxWriter::new();
        w.indent();
        w.line("ret;").unwrap();
        assert_eq!(w.finish(), "    ret;\n");
    }

    #[test]
    fn indent_dedent() {
        let mut w = PtxWriter::new();
        w.indent();
        w.indent();
        w.line("deep").unwrap();
        w.dedent();
        w.line("shallow").unwrap();
        w.dedent();
        w.line("top").unwrap();
        assert_eq!(w.finish(), "        deep\n    shallow\ntop\n");
    }

    #[test]
    fn dedent_saturates_at_zero() {
        let mut w = PtxWriter::new();
        w.dedent(); // should not panic
        w.line("ok").unwrap();
        assert_eq!(w.finish(), "ok\n");
    }

    #[test]
    fn blank_line() {
        let mut w = PtxWriter::new();
        w.line("a").unwrap();
        w.blank().unwrap();
        w.line("b").unwrap();
        assert_eq!(w.finish(), "a\n\nb\n");
    }

    #[test]
    fn instruction_formatting() {
        let mut w = PtxWriter::new();
        w.indent();
        let a = "%f0";
        let b = "%f1";
        let c = "%f2";
        w.instruction(".add.f32", &[&a, &b, &c]).unwrap();
        assert_eq!(w.finish(), "    .add.f32 %f0, %f1, %f2;\n");
    }

    #[test]
    fn instruction_no_operands() {
        let mut w = PtxWriter::new();
        w.indent();
        w.instruction("ret", &[]).unwrap();
        assert_eq!(w.finish(), "    ret;\n");
    }

    #[test]
    fn fresh_label_uniqueness() {
        let mut w = PtxWriter::new();
        let l0 = w.fresh_label("EXIT");
        let l1 = w.fresh_label("LOOP");
        let l2 = w.fresh_label("EXIT");
        assert_eq!(l0, "EXIT_0");
        assert_eq!(l1, "LOOP_1");
        assert_eq!(l2, "EXIT_2");
    }
}
