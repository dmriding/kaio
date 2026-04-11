//! PTX intermediate representation — the IR tree.
//!
//! A PTX program is modelled as a [`PtxModule`] containing one or more
//! [`PtxKernel`]s, each with parameters ([`PtxParam`]), a body of
//! [`PtxInstruction`]s, and a set of allocated [`Register`]s.

pub mod instruction;
pub mod kernel;
pub mod module;
pub mod operand;
pub mod param;
pub mod register;

pub use instruction::PtxInstruction;
pub use kernel::{KernelStats, PtxKernel, SharedDecl};
pub use module::PtxModule;
pub use operand::{Operand, SpecialReg};
pub use param::PtxParam;
pub use register::{Register, RegisterAllocator};
