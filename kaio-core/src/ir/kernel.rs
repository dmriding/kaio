//! PTX kernel — a single `.entry` function in a PTX module.

use super::instruction::PtxInstruction;
use super::param::PtxParam;
use super::register::Register;

/// Shared memory declaration in a PTX kernel preamble.
///
/// Emitted as `.shared .align {align} .b8 {name}[{size_bytes}];` after
/// register declarations.
#[derive(Debug, Clone)]
pub struct SharedDecl {
    /// Name of the shared memory allocation (e.g., `"sdata"`).
    pub name: String,
    /// Alignment in bytes (4 for f32, 8 for f64).
    pub align: u32,
    /// Total allocation size in bytes.
    pub size_bytes: u32,
}

/// A PTX kernel function (`.visible .entry`).
///
/// Built by constructing parameters, allocating registers, and pushing
/// instructions. Call [`set_registers`](Self::set_registers) with the
/// allocator's output before emission so the kernel knows which `.reg`
/// declarations to emit.
#[derive(Debug, Clone)]
pub struct PtxKernel {
    /// Kernel entry point name.
    pub name: String,
    /// Declared parameters (in signature order).
    pub params: Vec<PtxParam>,
    /// Instruction body.
    pub body: Vec<PtxInstruction>,
    /// All registers used, for `.reg` declaration emission.
    pub registers: Vec<Register>,
    /// Shared memory declarations (emitted after register declarations).
    pub shared_decls: Vec<SharedDecl>,
}

impl PtxKernel {
    /// Create a new empty kernel with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            params: Vec::new(),
            body: Vec::new(),
            registers: Vec::new(),
            shared_decls: Vec::new(),
        }
    }

    /// Add a parameter to the kernel signature.
    pub fn add_param(&mut self, param: PtxParam) {
        self.params.push(param);
    }

    /// Append an instruction to the kernel body.
    pub fn push(&mut self, instr: PtxInstruction) {
        self.body.push(instr);
    }

    /// Set the register list (from [`super::register::RegisterAllocator::into_allocated`]).
    pub fn set_registers(&mut self, regs: Vec<Register>) {
        self.registers = regs;
    }

    /// Add a shared memory declaration to the kernel preamble.
    pub fn add_shared_decl(&mut self, decl: SharedDecl) {
        self.shared_decls.push(decl);
    }
}
