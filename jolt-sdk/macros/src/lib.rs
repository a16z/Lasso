extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, PatType, ReturnType};

use common::constants::{INPUT_END_ADDRESS, INPUT_START_ADDRESS, OUTPUT_END_ADDRESS, OUTPUT_START_ADDRESS, PANIC_ADDRESS};

#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    let max_input_len = (INPUT_END_ADDRESS - INPUT_START_ADDRESS) as usize;

    let get_input_slice = quote! {
        let input_ptr = #INPUT_START_ADDRESS as *const u8;
        let input_slice = unsafe {
            core::slice::from_raw_parts(input_ptr, #max_input_len)
        };
    };

    let mut args = Vec::new();
    for arg in input_fn.sig.inputs {
        if let syn::FnArg::Typed(PatType { pat,ty, .. }) = arg {
            if let syn::Pat::Ident(pat_ident) = pat.as_ref() {

                let arg_name = &pat_ident.ident;
                let arg_type = &ty;

                let arg_fetch = quote! {
                    let x = input_slice[0];
                    let (#arg_name, input_slice) = jolt_sdk::postcard::take_from_bytes::<#arg_type>(input_slice).unwrap();
                };

                args.push(arg_fetch);
            } else {
                panic!("cannot parse arg");
            }
        } else {
            panic!("cannot parse arg");
        }
    }

    // TODO: ensure that input slice hasn't overflown
    let check_input_len = quote! {};

    let block = &input_fn.block;
    let block = quote! {let to_return = (|| -> _ { #block })();};

    let max_output_len = (OUTPUT_END_ADDRESS - OUTPUT_START_ADDRESS) as usize;
    let handle_return = match &input_fn.sig.output {
        ReturnType::Default => quote! {},
        ReturnType::Type(_, _) => quote! {
            let output_ptr = #OUTPUT_START_ADDRESS as *mut u8;
            let output_slice = unsafe {
                core::slice::from_raw_parts_mut(output_ptr, #max_output_len)
            };

            jolt_sdk::postcard::to_slice(&to_return, output_slice).unwrap();
        }
    };


    let transformed_fn = quote! {
        use core::arch::global_asm;
        use core::panic::PanicInfo;
        
        global_asm!("\
            .global _start\n\
            .extern _STACK_PTR\n\
            .section .text.boot\n\
            _start:	la sp, _STACK_PTR\n\
	            jal main\n\
	            j .\n\
        ");
        
        #[no_mangle]
        pub extern "C" fn main() {
            let mut offset = 0;
            #get_input_slice
            #(#args;)*
            #check_input_len
            #block
            #handle_return
        }
        
        #[panic_handler]
        fn panic(_info: &PanicInfo) -> ! {
            unsafe {
                core::ptr::write_volatile(#PANIC_ADDRESS as *mut u8, 1);
            }

            loop {}
        }
    };

    transformed_fn.into()
}
