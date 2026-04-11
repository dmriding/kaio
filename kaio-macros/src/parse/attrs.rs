//! Parse `#[gpu_kernel(block_size = N)]` or `#[gpu_kernel(block_size = (X, Y))]`
//! attribute arguments.

use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitInt, Token};

use crate::kernel_ir::KernelConfig;

/// Raw parsed attribute key-value pairs.
struct GpuKernelAttrs {
    block_size: Option<(u32, proc_macro2::Span)>,
    block_size_y: Option<(u32, proc_macro2::Span)>,
}

impl Parse for GpuKernelAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut block_size = None;
        let mut block_size_y = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "block_size" => {
                    if input.peek(syn::token::Paren) {
                        // 2D: block_size = (X, Y)
                        let content;
                        let _paren = syn::parenthesized!(content in input);
                        let x_lit: LitInt = content.parse()?;
                        content.parse::<Token![,]>()?;
                        let y_lit: LitInt = content.parse()?;
                        let span = x_lit.span();
                        let x_val: u32 = x_lit.base10_parse().map_err(|_| {
                            syn::Error::new(span, "block_size X must be a positive integer")
                        })?;
                        let y_val: u32 = y_lit.base10_parse().map_err(|_| {
                            syn::Error::new(
                                y_lit.span(),
                                "block_size Y must be a positive integer",
                            )
                        })?;
                        block_size = Some((x_val, span));
                        block_size_y = Some((y_val, y_lit.span()));
                    } else {
                        // 1D: block_size = N
                        let lit: LitInt = input.parse()?;
                        let span = lit.span();
                        let value: u32 = lit.base10_parse().map_err(|_| {
                            syn::Error::new(span, "block_size must be a positive integer")
                        })?;
                        block_size = Some((value, span));
                    }
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

        Ok(GpuKernelAttrs {
            block_size,
            block_size_y,
        })
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

    if let Some((block_size_y, y_span)) = attrs.block_size_y {
        // 2D validation
        if block_size == 0 {
            return Err(syn::Error::new(
                block_size_span,
                "block_size X dimension must be > 0",
            ));
        }
        if block_size_y == 0 {
            return Err(syn::Error::new(y_span, "block_size Y dimension must be > 0"));
        }
        let total = block_size * block_size_y;
        if total > 1024 {
            return Err(syn::Error::new(
                block_size_span,
                format!(
                    "total thread count ({block_size} * {block_size_y} = {total}) \
                     cannot exceed 1024"
                ),
            ));
        }
        Ok(KernelConfig {
            block_size,
            block_size_y: Some(block_size_y),
            block_size_span,
        })
    } else {
        // 1D validation (unchanged)
        if !block_size.is_power_of_two() {
            return Err(syn::Error::new(
                block_size_span,
                format!("`block_size` must be a power of 2, got {block_size}"),
            ));
        }
        if block_size > 1024 {
            return Err(syn::Error::new(
                block_size_span,
                format!("`block_size` cannot exceed 1024, got {block_size}"),
            ));
        }
        Ok(KernelConfig {
            block_size,
            block_size_y: None,
            block_size_span,
        })
    }
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
        assert_eq!(config.block_size_y, None);
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

    // --- 2D block_size tests ---

    #[test]
    fn parse_block_size_2d() {
        let tokens = quote! { block_size = (16, 16) };
        let config = parse_kernel_config(tokens).unwrap();
        assert_eq!(config.block_size, 16);
        assert_eq!(config.block_size_y, Some(16));
    }

    #[test]
    fn parse_block_size_2d_asymmetric() {
        let tokens = quote! { block_size = (32, 8) };
        let config = parse_kernel_config(tokens).unwrap();
        assert_eq!(config.block_size, 32);
        assert_eq!(config.block_size_y, Some(8));
    }

    #[test]
    fn reject_block_size_2d_exceeds_1024() {
        let tokens = quote! { block_size = (32, 64) };
        let err = parse_kernel_config(tokens).unwrap_err();
        assert!(err.to_string().contains("1024"));
    }

    #[test]
    fn reject_block_size_2d_zero_y() {
        let tokens = quote! { block_size = (16, 0) };
        let err = parse_kernel_config(tokens).unwrap_err();
        assert!(err.to_string().contains("must be > 0"));
    }
}
