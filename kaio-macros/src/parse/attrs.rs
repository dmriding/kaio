//! Parse `#[gpu_kernel(block_size = N)]` attribute arguments.

use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitInt, Token};

use crate::kernel_ir::KernelConfig;

/// Raw parsed attribute key-value pairs.
struct GpuKernelAttrs {
    block_size: Option<(u32, proc_macro2::Span)>,
}

impl Parse for GpuKernelAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut block_size = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "block_size" => {
                    let lit: LitInt = input.parse()?;
                    let span = lit.span();
                    let value: u32 = lit.base10_parse().map_err(|_| {
                        syn::Error::new(span, "block_size must be a positive integer")
                    })?;
                    block_size = Some((value, span));
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown attribute `{other}`. Expected: block_size"),
                    ));
                }
            }

            // Consume trailing comma if present
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(GpuKernelAttrs { block_size })
    }
}

/// Parse the attribute token stream into a validated `KernelConfig`.
pub fn parse_kernel_config(attr: TokenStream) -> syn::Result<KernelConfig> {
    let attrs: GpuKernelAttrs = syn::parse2(attr)?;

    let (block_size, block_size_span) = attrs.block_size.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "`block_size` is required in `#[gpu_kernel(...)]`",
        )
    })?;

    // Validate: must be power of 2
    if !block_size.is_power_of_two() {
        return Err(syn::Error::new(
            block_size_span,
            format!("`block_size` must be a power of 2, got {block_size}"),
        ));
    }

    // Validate: must be in range [1, 1024]
    if block_size > 1024 {
        return Err(syn::Error::new(
            block_size_span,
            format!("`block_size` cannot exceed 1024, got {block_size}"),
        ));
    }

    Ok(KernelConfig {
        block_size,
        block_size_span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn parse_block_size_256() {
        let tokens = quote! { block_size = 256 };
        let config = parse_kernel_config(tokens).unwrap();
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn parse_block_size_1() {
        let tokens = quote! { block_size = 1 };
        let config = parse_kernel_config(tokens).unwrap();
        assert_eq!(config.block_size, 1);
    }

    #[test]
    fn parse_block_size_1024() {
        let tokens = quote! { block_size = 1024 };
        let config = parse_kernel_config(tokens).unwrap();
        assert_eq!(config.block_size, 1024);
    }

    #[test]
    fn parse_block_size_with_trailing_comma() {
        let tokens = quote! { block_size = 128, };
        let config = parse_kernel_config(tokens).unwrap();
        assert_eq!(config.block_size, 128);
    }

    #[test]
    fn reject_missing_block_size() {
        let tokens = quote! {};
        let err = parse_kernel_config(tokens).unwrap_err();
        assert!(err.to_string().contains("block_size"));
    }

    #[test]
    fn reject_non_power_of_two() {
        let tokens = quote! { block_size = 100 };
        let err = parse_kernel_config(tokens).unwrap_err();
        assert!(err.to_string().contains("power of 2"));
    }

    #[test]
    fn reject_exceeds_1024() {
        let tokens = quote! { block_size = 2048 };
        let err = parse_kernel_config(tokens).unwrap_err();
        assert!(err.to_string().contains("1024"));
    }

    #[test]
    fn reject_unknown_attribute() {
        let tokens = quote! { block_size = 256, warp_size = 32 };
        let err = parse_kernel_config(tokens).unwrap_err();
        assert!(err.to_string().contains("unknown attribute"));
    }
}
