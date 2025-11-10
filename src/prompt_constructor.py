import os
from .utils import read_file
from prompt_generator import *

"""
Multi-Language Prompt Constructor

Supports: Triton, CuTe (TileLang currently disabled/commented out)

Design principles: 
- To evaluate base model performance on KernelBench, we use the simplest prompt possible to guide model output to generated desired output format.
- However, we do not do extensive prompt engineering or few-shot examples in the LLM to steer behaviour. 
"""
def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)


def get_arch_definition(arch_src):
    """
    Construct torch definition from original torch nn.Module definition
    """
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


############################################
# CUDA Prompt
############################################
@register_prompt("CUDA")
class CUDAPromptGenerator(BackendPromptGenerator):
    def prompt_generate_custom_cuda_fewshot_and_template(self, ref_arch_src: str, shots: list) -> str:
        """
        Generate a prompt with specified few-shot examples following a template 

        shots: list of few-shot examples to include in the prompt
        Avaliable few shot options to start with: 
        - ex_add: pointwise addition
        - ex_fuse_gelu: fused gelu
        - ex_mnist2: fused convolutions and relus (DEPRECATED)
        - ex_tiled_matmul: tiled matrix multiplication
        - ex_flash_attn: simple flash attention
        """
        prompt = self.problem_statement_cleaned

        # k = 1
        example_add = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_add.py")
        )
        example_add_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_add.py")
        )
        example_add_desc = "This given architecture is for a pointwise addition: "

        # k = 2
        example_fuse_gelu = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_fuse_gelu.py")
        )
        example_fuse_gelu_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_fuse_gelu.py")
        )
        example_fuse_gelu_desc = "This given architecture is for a fused gelu: "

        # k = 3 (DEPRECATED)
        example_mnist2 = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_mnist2.py")
        )
        example_mnist2_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_mnist2.py")
        )
        exmaple_mnist2_desc = "This given architecture is for a model with fused convolutions and relus: "

        # k = 4
        example_tiled_matmul = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_tiled_matmul.py")
        )
        example_tiled_matmul_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_tiled_matmul.py")
        )
        example_tiled_matmul_desc = "This given architecture is for a model with tiled matrix multiplication: "

        # k = 5
        example_flash_attn = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_flash_attn.py")
        )
        example_flash_attn_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_flash_attn.py")
        )
        example_flash_attn_desc = "This given architecture is for a model with simple io-aware implementation of attention, also known as flash attention: "

        examples = []
        for s in shots:
            if s not in ["ex_add", "ex_fuse_gelu", "ex_mnist2", "ex_tiled_matmul", "ex_flash_attn"]:
                raise ValueError(f"Invalid shot: {s}")
            elif s == "ex_add":
                examples.append((example_add, example_add_new, example_add_desc))
            elif s == "ex_fuse_gelu":
                examples.append((example_fuse_gelu, example_fuse_gelu_new, example_fuse_gelu_desc))
            elif s == "ex_mnist2": # DEPRECATED
                raise ValueError("ex_mnist2 is deprecated")
                examples.append((example_mnist2, example_mnist2_new, exmaple_mnist2_desc))
            elif s == "ex_tiled_matmul":
                examples.append((example_tiled_matmul, example_tiled_matmul_new, example_tiled_matmul_desc))
            elif s == "ex_flash_attn":
                examples.append((example_flash_attn, example_flash_attn_new, example_flash_attn_desc))
        

        for i, tup in enumerate(examples):
            base, kernel, desc = tup

            prompt += f"""
    Example {i+1}:\n\n
    Here is an example architecture:\n\n
    ```
    {base}
    ```\n
    {self.problem_statement_cleaned} \n
    Here is an optimized verison with custom CUDA kernels: \n
    ```
    {kernel}
    ```\n\n
    """

    # should we put task here?
        prompt += f"""
    Task:\n\n
    Here is an example architecture:\n\n
    ```
    {ref_arch_src}
    ```\n
    """
        prompt += self.problem_statement_cleaned
        return prompt

    def prompt_generate_ex_with_CoT_template(self, ref_arch_src: str, cot_example: str) -> str:
        """
        Generate a prompt with a CoT example following a template 
        Avaliable CoT examples: 
        - ex_fuse_gelu: fused gelu
        - ex_mnist2: fused convolutions and relus
        - ex_tiled_matmul: tiled matrix multiplication
        """

        # I updated this to allow CoT. Also explicilty state think step by step.
        PROBLEM_INSTRUCTION_COT = """
    Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Do not output testing code. 
    In the end, make sure the final code block contains code for output architecture ModelNew with cuda code.\n
    Let's think step by step.\n
    """ 

        prompt = self.problem_statement_cleaned
        
        assert cot_example in ["ex_fuse_gelu", "ex_mnist2", "ex_tiled_matmul"]

        # k = 2
        example_fuse_gelu = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_fuse_gelu.py")
        )
        example_fuse_gelu_cot = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_fuse_gelu.py")
        )
        example_fuse_gelu_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_fuse_gelu.py")
        )
        example_fuse_gelu_desc = "This given architecture is for a fused gelu: "

        # k = 3
        example_mnist2 = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_mnist2.py")
        )
        example_mnist2_cot = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_mnist2.py")
        )
        example_mnist2_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_mnist2.py")
        )
        exmaple_mnist2_desc = "This given architecture is for a model with fused convolutions and relus: "

        # k = 4
        example_tiled_matmul = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_ex_tiled_matmul.py")
        )
        example_tiled_matmul_cot = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/cot/model_cot_tiled_matmul.py")
        )
        example_tiled_matmul_new = read_file(
            os.path.join(REPO_TOP_PATH, "src/prompts/few_shot/model_new_ex_tiled_matmul.py")
        )
        example_tiled_matmul_desc = "This given architecture is for a model with tiled matrix multiplication: "
        
        match cot_example:
            case "ex_fuse_gelu":
                base = example_fuse_gelu
                cot = example_fuse_gelu_cot
                kernel = example_fuse_gelu_new
                desc = example_fuse_gelu_desc
            case "ex_mnist2":
                base = example_mnist2
                cot = example_mnist2_cot
                kernel = example_mnist2_new
                desc = exmaple_mnist2_desc
            case "ex_tiled_matmul":
                base = example_tiled_matmul
                cot = example_tiled_matmul_cot
                kernel = example_tiled_matmul_new
                desc = example_tiled_matmul_desc
            case _:
                raise ValueError(f"Invalid CoT example: {cot_example} not found in CoT examples")

        # construct example with 
        # NOTE: we only do one example with CoT for now
        # 1. ref_src problem -> 2. Instruction -> 3. CoT -> 4. Solution
        prompt += f"""
    Here is an example architecture:\n\n
    ```
    {base}
    ```\n
    {PROBLEM_INSTRUCTION_COT} \n
    {cot} \n
    ```
    {kernel}
    ```\n\n
    """

    # show task to solve
        prompt += f"""
    Task:\n\n
    Here is an example architecture:\n\n
    ```
    {ref_arch_src}
    ```\n
    """
        prompt += PROBLEM_INSTRUCTION_COT

        return prompt


    def prompt_generate_prompt_with_hardware_info_from_template(self, ref_arch_src: str, gpu_name: str) -> str:
        """
        Similar to prompt_generate_custom_cuda_from_prompt_template, 
        but with hardware information for the given GPU
        """

        arch = ref_arch_src
        # These are strictly defined for now

        # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
        example_arch_path = os.path.join(
            REPO_TOP_PATH, f"src/prompts/model_ex_add.py"
        )
        example_new_arch_path = os.path.join(
            REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py"
        )

        gpu_spec_file_path = os.path.join(REPO_TOP_PATH, f"src/prompts/hardware/gpu_specs.py")

        example_arch = read_file(example_arch_path)
        example_new_arch = read_file(example_new_arch_path)
        gpu_spec_info = read_file(gpu_spec_file_path)

        return self.prompt_generate_prompt_with_hardware_info(
                                            ref_arch_src=arch, 
                                            gpu_name=gpu_name, 
                                            example_arch_src=example_arch, 
                                            example_new_arch_src=example_new_arch, 
                                            gpu_spec_info_src=gpu_spec_info
                                            )
        


    def prompt_generate_prompt_with_hardware_info(self, ref_arch_src: str, 
                                                gpu_name: str, 
                                                example_arch_src: str, 
                                                example_new_arch_src: str, 
                                                gpu_spec_info_src: str) -> str:
        """
        Generate a prompt with hardware information for the given GPU
        gpu_spec_info_src: str of the gpu spec src file
        """

        # Create a dictionary to store the local namespace
        local_dict = {}
        
        # Execute the GPU spec file in the local namespace
        exec(gpu_spec_info_src, {}, local_dict)
        
        # Get the required variables from the local namespace
        GPU_SPEC_INFO = local_dict.get('GPU_SPEC_INFO')
        GPU_DEFINITIONS = local_dict.get('GPU_DEFINITIONS')
        GPU_BEST_PRACTICES = local_dict.get('GPU_BEST_PRACTICES')
        
        if not GPU_SPEC_INFO or not GPU_DEFINITIONS or not GPU_BEST_PRACTICES:
            raise ValueError("GPU_SPEC_INFO or GPU_DEFINITIONS or GPU_BEST_PRACTICES not found in gpu_spec_info_src")

        assert gpu_name in GPU_SPEC_INFO, f"GPU name {gpu_name} not found in GPU_SPEC_INFO"

        prompt = self.problem_statement

        if example_arch_src != "" and example_new_arch_src != "":
            prompt += f"""
            Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
            ``` \n
            {example_arch_src}
            ``` \n
            The example new arch with custom CUDA kernels looks like this: 
            ```
            {example_new_arch_src}
            ``` \n
            """
        
        curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]

        gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
        prompt += f"""
        Here is some information about the underlying hardware that you should keep in mind. \n\n
    The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.\n\n"""
        
        for key, value in curr_gpu_spec_info.items():
            if key == "GPU Architecture":
                continue
            prompt += f"""- We have {value} of {key}.\n"""
        
        
        prompt += f"""\n\n
    Here are some concepts about the GPU architecture that could be helpful: \n\n"""
        for key, value in GPU_DEFINITIONS.items():
            prompt += f"""- {key}: {value}\n"""

        prompt += f"""\n\n
    Here are some best practices for writing CUDA kernels on GPU: \n\n"""
        for best_practice in GPU_BEST_PRACTICES:
            prompt += f"""- {best_practice}\n"""


        prompt += f"""
        You are given the following architecture: \n
        ```
        {ref_arch_src}
        ```
        """
        

        prompt += self.instruction
        return prompt

################################################################################
# Triton Backend
################################################################################
@register_prompt("Triton")
class TritonPromptGenerator(BackendPromptGenerator):
    def __init__(self):
        super().__init__("Triton")

    def prompt_generate_custom_triton_fewshot_and_template(
        self, ref_arch_src: str, shots: list
    ) -> str:
        raise NotImplementedError("This function has not been implemented yet")


    def prompt_generate_ex_with_CoT_template_triton(self, ref_arch_src: str, cot_example: str) -> str:
        raise NotImplementedError("This function has not been implemented yet")


    def prompt_generate_prompt_with_hardware_info_from_template_triton(
        self, ref_arch_src: str, gpu_name: str
    ) -> str:
        """
        Similar to prompt_generate_custom_triton_from_prompt_template,
        but with hardware information for the given GPU
        """
        arch = ref_arch_src

        example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
        example_new_arch_path = os.path.join(
            REPO_TOP_PATH, f"src/prompts/model_new_ex_add_triton.py"
        )
        gpu_spec_file_path = os.path.join(
            REPO_TOP_PATH, f"src/prompts/hardware/gpu_specs.py"
        )

        example_arch = read_file(example_arch_path)
        example_new_arch = read_file(example_new_arch_path)
        gpu_spec_info = read_file(gpu_spec_file_path)

        return self.prompt_generate_prompt_with_hardware_info_triton(
            ref_arch_src=arch,
            gpu_name=gpu_name,
            example_arch_src=example_arch,
            example_new_arch_src=example_new_arch,
            gpu_spec_info_src=gpu_spec_info,
        )


    def prompt_generate_prompt_with_hardware_info_triton(
        self,
        ref_arch_src: str,
        gpu_name: str,
        example_arch_src: str,
        example_new_arch_src: str,
        gpu_spec_info_src: str,
    ) -> str:
        """
        Generate a prompt with hardware information for the given GPU
        gpu_spec_info_src: str of the gpu spec src file
        """
        local_dict = {}
        exec(gpu_spec_info_src, {}, local_dict)

        GPU_SPEC_INFO = local_dict.get("GPU_SPEC_INFO")
        GPU_DEFINITIONS = local_dict.get("GPU_DEFINITIONS")
        GPU_BEST_PRACTICES = local_dict.get("GPU_BEST_PRACTICES")

        if not GPU_SPEC_INFO or not GPU_DEFINITIONS or not GPU_BEST_PRACTICES:
            raise ValueError(
                "GPU_SPEC_INFO or GPU_DEFINITIONS or GPU_BEST_PRACTICES not found in gpu_spec_info_src"
            )

        assert gpu_name in GPU_SPEC_INFO, f"GPU name {gpu_name} not found in GPU_SPEC_INFO"

        prompt = self.problem_statement

        if example_arch_src != "" and example_new_arch_src != "":
            prompt += f"""
            Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: \n
            ``` \n
            {example_arch_src}
            ``` \n
            The example new arch with custom Triton kernels looks like this: 
            ```
            {example_new_arch_src}
            ``` \n
            """

        curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]
        gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
        prompt += f"""
        Here is some information about the underlying hardware that you should keep in mind. \n\n
    The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.\n\n"""

        for key, value in curr_gpu_spec_info.items():
            if key == "GPU Architecture":
                continue
            prompt += f"""- We have {value} of {key}.\n"""

        prompt += f"""\n\n
    Here are some concepts about the GPU architecture that could be helpful: \n\n"""
        for key, value in GPU_DEFINITIONS.items():
            prompt += f"""- {key}: {value}\n"""

        prompt += f"""\n\n
    Here are some best practices for writing Triton kernels on GPU: \n\n"""
        for best_practice in GPU_BEST_PRACTICES:
            prompt += f"""- {best_practice}\n"""

        prompt += f"""
        You are given the following architecture: \n
        ```
        {ref_arch_src}
        ```
        """

        prompt += self.instruction
        return prompt



################################################################################
# TileLang Backend
################################################################################
@register_prompt("TileLang")
class TileLangPromptGenerator(BackendPromptGenerator):
    def __init__(self):
        super().__init__("TileLang")


################################################################################
# CuTe Backend
################################################################################
@register_prompt("CuTe")
class CuTePromptGenerator(BackendPromptGenerator):
    def __init__(self):
        super().__init__("CuTe")

################################################################################
# Tenstorrent PyKernel Backend
################################################################################
@register_prompt("PyKernel")
class TTPyKernelPromptGenerator(BackendPromptGenerator):
    def __init__(self):
        super().__init__("PyKernel")

################################################################################
# Unified API
################################################################################

def get_prompt_for_backend(ref_arch_src: str, backend: str = "cuda") -> str:
    """
    Unified API to get prompt for any supported backend
    
    Args:
        ref_arch_src: Reference architecture source code
        backend: One of 'triton', 'tilelang', 'cute'
    
    Returns:
        Prompt string for the specified backend
    """
    backend_lower = backend.lower()
    if backend_lower in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[backend_lower].prompt_generate_custom_from_prompt_template(ref_arch_src)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. Must be one of: {list(PROMPT_REGISTRY.keys())}"
        )


################################################################################
# Main (for testing)
################################################################################

def main():
    gpu_name = "L40S"
    backend = "cuda"  # Change this to test different backends
    KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

    ref_arch_src = read_file(os.path.join(KERNEL_BENCH_PATH, f"level1/19_ReLU.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"
    
    prompt = get_prompt_for_backend(ref_arch_src, backend)
    print(f"\n{'='*80}\n{backend.upper()} PROMPT:\n{'='*80}\n")
    print(prompt)
    
    # Write prompt to temp file
    temp_file_path = os.path.join(REPO_TOP_PATH, "scratch", f"prompt_{backend}_draft.txt")
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        f.write(prompt)
    print(f"\nPrompt written to: {temp_file_path}")

if __name__ == "__main__":
    main()



