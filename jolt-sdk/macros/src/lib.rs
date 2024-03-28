extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_macro_input, AttributeArgs, Ident, ItemFn, Meta, MetaNameValue, NestedMeta, PatType,
    ReturnType, Type,
};

use common::constants::{
    INPUT_END_ADDRESS, INPUT_START_ADDRESS, OUTPUT_END_ADDRESS, OUTPUT_START_ADDRESS, PANIC_ADDRESS,
};

#[proc_macro_attribute]
pub fn main(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as AttributeArgs);
    let func = parse_macro_input!(item as ItemFn);
    MacroBuilder::new(attr, func).build()
}

struct MacroBuilder {
    attr: AttributeArgs,
    func: ItemFn,
    func_args: Vec<(Ident, Box<Type>)>,
}

impl MacroBuilder {
    fn new(attr: AttributeArgs, func: ItemFn) -> Self {
        let func_args = Self::get_func_args(&func);
        Self {
            attr,
            func,
            func_args,
        }
    }

    fn build(&self) -> TokenStream {
        let build_fn = self.make_build_fn();
        let execute_fn = self.make_execute_function();
        let analyze_fn = self.make_analyze_function();
        let preprocess_fn = self.make_preprocess_func();
        let prove_fn = self.make_prove_func();
        let main_fn = self.make_main_func();

        quote! {
            #build_fn
            #execute_fn
            #analyze_fn
            #preprocess_fn
            #prove_fn
            #main_fn
        }
        .into()
    }

    fn make_build_fn(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let build_fn_name = Ident::new(&format!("build_{}", fn_name), fn_name.span());
        let prove_output_ty = self.get_prove_output_type();

        let input_names = self.func_args.iter().map(|(name, _)| name);
        let input_types = self.func_args.iter().map(|(_, ty)| ty);
        let inputs = &self.func.sig.inputs;
        let preprocess_fn_name = Ident::new(&format!("preprocess_{}", fn_name), fn_name.span());
        let prove_fn_name = Ident::new(&format!("prove_{}", fn_name), fn_name.span());
        let imports = self.make_imports();

        quote! {
            #[cfg(not(feature = "guest"))]
            pub fn #build_fn_name() -> (
                impl Fn(#(#input_types),*) -> #prove_output_ty,
                impl Fn(jolt_sdk::Proof) -> bool
            ) {
                #imports
                let (program, preprocessing) = #preprocess_fn_name();
                let program = std::rc::Rc::new(program);
                let preprocessing = std::rc::Rc::new(preprocessing);

                let program_cp = program.clone();
                let preprocessing_cp = preprocessing.clone();

                let prove_closure = move |#inputs| {
                    let program = (*program).clone();
                    let preprocessing = (*preprocessing).clone();
                    #prove_fn_name(program, preprocessing, #(#input_names),*)
                };


                let verify_closure = move |proof: jolt_sdk::Proof| {
                    let program = (*program_cp).clone();
                    let preprocessing = (*preprocessing_cp).clone();
                    RV32IJoltVM::verify(preprocessing, proof.proof, proof.commitments).is_ok()
                };

                (prove_closure, verify_closure)
            }
        }
    }

    fn make_execute_function(&self) -> TokenStream2 {
        let fn_name = self.get_func_name();
        let execute_fn_name = Ident::new(&format!("execute_{}", fn_name), fn_name.span());
        let inputs = &self.func.sig.inputs;
        let output = &self.func.sig.output;
        let body = &self.func.block;

        quote! {
             #[cfg(not(feature = "guest"))]
             pub fn #execute_fn_name(#inputs) #output {
                 #body
             }
        }
    }

    fn make_analyze_function(&self) -> TokenStream2 {
        let set_mem_size = self.make_set_memory_size();
        let guest_name = self.get_guest_name();
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let analyze_fn_name = Ident::new(&format!("analyze_{}", fn_name), fn_name.span());
        let inputs = &self.func.sig.inputs;
        let set_program_args = self.func_args.iter().map(|(name, _)| {
            quote! {
                program.set_input(&#name);
            }
        });

        quote! {
             #[cfg(not(feature = "guest"))]
             pub fn #analyze_fn_name(#inputs) {
                #imports

                let mut program = Program::new(#guest_name);
                #set_mem_size
                #(#set_program_args;)*

                program.trace_analyze();
             }
        }
    }

    fn make_preprocess_func(&self) -> TokenStream2 {
        let set_mem_size = self.make_set_memory_size();
        let guest_name = self.get_guest_name();
        let imports = self.make_imports();

        let fn_name = self.get_func_name();
        let preprocess_fn_name = Ident::new(&format!("preprocess_{}", fn_name), fn_name.span());
        quote! {
            #[cfg(not(feature = "guest"))]
            pub fn #preprocess_fn_name() -> (
                jolt_sdk::host::Program,
                jolt_sdk::JoltPreprocessing<jolt_sdk::F, jolt_sdk::G>
            ) {
                #imports

                let mut program = Program::new(#guest_name);
                #set_mem_size
                let bytecode = program.decode();

                // TODO(moodlezoup): Feed in size parameters via macro
                let preprocessing: JoltPreprocessing<jolt_sdk::F, jolt_sdk::G> =
                    RV32IJoltVM::preprocess(
                        bytecode,
                        1 << 20,
                        1 << 20,
                        1 << 24
                    );

                (program, preprocessing)
            }
        }
    }

    fn make_prove_func(&self) -> TokenStream2 {
        let prove_output_ty = self.get_prove_output_type();

        let handle_return = match &self.func.sig.output {
            ReturnType::Default => quote! {
                let ret_val = ();
            },
            ReturnType::Type(_, ty) => quote! {
                let ret_val = jolt_sdk::postcard::from_bytes::<#ty>(&output_bytes).unwrap();
            },
        };

        let set_program_args = self.func_args.iter().map(|(name, _)| {
            quote! {
                program.set_input(&#name);
            }
        });

        let fn_name = self.get_func_name();
        let inputs = &self.func.sig.inputs;
        let imports = self.make_imports();

        let prove_fn_name = syn::Ident::new(&format!("prove_{}", fn_name), fn_name.span());
        quote! {
            #[cfg(not(feature = "guest"))]
            pub fn #prove_fn_name(
                mut program: jolt_sdk::host::Program,
                preprocessing: jolt_sdk::JoltPreprocessing<jolt_sdk::F, jolt_sdk::G>,
                #inputs
            ) -> #prove_output_ty {
                #imports

                #(#set_program_args;)*

                let bytecode = program.decode();
                let (io_device, bytecode_trace, instruction_trace, memory_trace, circuit_flags) =
                    program.trace();

                let output_bytes = io_device.outputs.clone();

                let (jolt_proof, jolt_commitments) = RV32IJoltVM::prove(
                    io_device,
                    bytecode,
                    bytecode_trace,
                    memory_trace,
                    instruction_trace,
                    circuit_flags,
                    preprocessing,
                );

                #handle_return

                let proof = jolt_sdk::Proof {
                    proof: jolt_proof,
                    commitments: jolt_commitments,
                };

                (ret_val, proof)
            }
        }
    }

    fn make_main_func(&self) -> TokenStream2 {
        let max_input_len = (INPUT_END_ADDRESS - INPUT_START_ADDRESS) as usize;

        let get_input_slice = quote! {
            let input_ptr = #INPUT_START_ADDRESS as *const u8;
            let input_slice = unsafe {
                core::slice::from_raw_parts(input_ptr, #max_input_len)
            };
        };

        let args = &self.func_args;
        let args_fetch = args.iter().map(|(name, ty)| {
            quote! {
                let (#name, input_slice) =
                    jolt_sdk::postcard::take_from_bytes::<#ty>(input_slice).unwrap();
            }
        });

        // TODO: ensure that input slice hasn't overflown
        let check_input_len = quote! {};

        let block = &self.func.block;
        let block = quote! {let to_return = (|| -> _ { #block })();};

        let max_output_len = (OUTPUT_END_ADDRESS - OUTPUT_START_ADDRESS) as usize;
        let handle_return = match &self.func.sig.output {
            ReturnType::Default => quote! {},
            ReturnType::Type(_, _) => quote! {
                let output_ptr = #OUTPUT_START_ADDRESS as *mut u8;
                let output_slice = unsafe {
                    core::slice::from_raw_parts_mut(output_ptr, #max_output_len)
                };

                jolt_sdk::postcard::to_slice(&to_return, output_slice).unwrap();
            },
        };

        quote! {
            #[cfg(feature = "guest")]
            use core::arch::global_asm;
            #[cfg(feature = "guest")]
            use core::panic::PanicInfo;

            #[cfg(feature = "guest")]
            global_asm!("\
                .global _start\n\
                .extern _STACK_PTR\n\
                .section .text.boot\n\
                _start:	la sp, _STACK_PTR\n\
                    jal main\n\
                    j .\n\
            ");

            #[cfg(feature = "guest")]
            #[no_mangle]
            pub extern "C" fn main() {
                let mut offset = 0;
                #get_input_slice
                #(#args_fetch;)*
                #check_input_len
                #block
                #handle_return
            }

            #[cfg(feature = "guest")]
            #[panic_handler]
            fn panic(_info: &PanicInfo) -> ! {
                unsafe {
                    core::ptr::write_volatile(#PANIC_ADDRESS as *mut u8, 1);
                }

                loop {}
            }
        }
    }

    fn make_imports(&self) -> TokenStream2 {
        quote! {
            #[cfg(not(feature = "guest"))]
            use jolt_sdk::{
                CurveGroup,
                PrimeField,
                host::Program,
                JoltPreprocessing,
                Jolt,
                JoltCommitments,
                RV32IJoltVM,
                RV32I,
                RV32IJoltProof,
                BytecodeRow,
                MemoryOp,
                MEMORY_OPS_PER_INSTRUCTION,
                instruction::add::ADDInstruction,
                tracer,
            };
        }
    }

    fn make_set_memory_size(&self) -> TokenStream2 {
        if self.attr.len() == 1 {
            match &self.attr[0] {
                NestedMeta::Meta(Meta::NameValue(MetaNameValue { path, lit, .. })) => {
                    let ident = &path.get_ident().expect("Expected identifier");
                    if ident.to_string() != "memory_size" {
                        panic!("only allowed attribute is memory_size");
                    }

                    quote! {
                        program.set_memory_size(#lit);
                    }
                }
                _ => panic!("expected integer literal"),
            }
        } else if self.attr.len() > 1 {
            panic!("only one attribute expected");
        } else {
            quote!()
        }
    }

    fn get_prove_output_type(&self) -> TokenStream2 {
        match &self.func.sig.output {
            ReturnType::Default => quote! {
                ((), jolt_sdk::Proof)
            },
            ReturnType::Type(_, ty) => quote! {
                (#ty, jolt_sdk::Proof)
            },
        }
    }

    fn get_func_args(func: &ItemFn) -> Vec<(Ident, Box<Type>)> {
        let mut args = Vec::new();
        for arg in &func.sig.inputs {
            if let syn::FnArg::Typed(PatType { pat, ty, .. }) = arg {
                if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                    args.push((pat_ident.ident.clone(), ty.clone()));
                } else {
                    panic!("cannot parse arg");
                }
            } else {
                panic!("cannot parse arg");
            }
        }

        args
    }

    fn get_func_name(&self) -> &Ident {
        &self.func.sig.ident
    }

    fn get_guest_name(&self) -> String {
        std::env::var("CARGO_PKG_NAME").unwrap()
    }
}
