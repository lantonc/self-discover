from . import config
from enum import Enum
from dataclasses import dataclass, field
from . import utils
# from . import gemini 
import os
# from . import localLLM
from .utils import log_print
import json
from . import openai

__all__ = ['ModelType','LLMConfig','select','adapt','implement','solve','self_discover','ChatTemplate']


class ModelType(Enum):
    GEMINI = "gemini"
    LOCAL = "local"
    OPENAI = "openai"

class ChatTemplate(Enum):
    MIXTRAL_INSTRUCT = "mixtral_instruct"
    ZEPHYR = "zephyr"
    CHATML = "chatml"
    
    

@dataclass
class LLMConfig:
    gguf_path: str = field(default=None,metadata={"description": "Only for local ModelType. Path to GGUF file."})
    context_length: int = field(default=2000,metadata={"description": "Context Limit in terms of Tokens."})
    threads: int = field(default=12,metadata={"description": "Only for local ModelType. Number of threads to use."})
    api_key: str = field(default=None,metadata={"description": "Not used by LOCAL ModelTypes. API key used for model."})
    temp: float = field(default=0.8,metadata={"description": "LLM Temperature setting."})
    chat_template: ChatTemplate = field(default=ChatTemplate.MIXTRAL_INSTRUCT,metadata={"description": "Only for local ModelType. Chat Template Type"})
    model_type: ModelType = field(default=ModelType.GEMINI,metadata={"description": "Model Type."})
    model_name: str = field(default=None,metadata={"description": "Model name. Only used by Gemini or OPENAI models. Use this to set the model to call, for example 'gpt-3.5-turbo-0125' if using openai models. "})
    
    
def formatPrompt(prompt: str,llmConfig: LLMConfig) ->tuple:
    stop = []
    if llmConfig.chat_template == ChatTemplate.MIXTRAL_INSTRUCT:
        prompt = f"[INST] {prompt} [/INST]"
        stop = ["[/INST]"]
    elif llmConfig.chat_template == ChatTemplate.ZEPHYR:
        prompt = f"<|system|>\nYou are a friendly AI assistant who follows directions.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        stop = ["</s>"]
    elif llmConfig.chat_template == ChatTemplate.CHATML:
        prompt = f"<|im_start|>system \nYou are a helpful AI chatbot who pays very close attention to instructions from the user - especially any instructions on how to format your response.<|im_end|><|im_start|>user\n {prompt}<|im_end|>\n<|im_start|>assistant\n"
        stop = ['<|im_end|>']
        
    else:
        raise ValueError("Invalid ChatTemplate Enum!")
    
    return (prompt,stop)


def select(task: str, llmConfig: LLMConfig = LLMConfig(), print_in_out:bool = False) -> dict:
    
    selection = None
    
    if llmConfig.model_type == ModelType.OPENAI:
        selection = __select_openai(llmConfig=llmConfig,task=task, print_in_out=print_in_out)
    else:
        raise ValueError("Unsupported model Type!")
    
    return selection

def adapt(task: str, reasoning_modules: dict, llmConfig: LLMConfig = LLMConfig(), print_in_out:bool = False) -> str:
    
    adapted_modules = None
    
    if llmConfig.model_type == ModelType.OPENAI:
        adapted_modules = __adapt_openai(llmConfig=llmConfig,task=task,reasoning_modules=reasoning_modules, print_in_out=print_in_out)
    else: 
        raise ValueError("Unsupported model Type!")
    
    return adapted_modules

def implement(task: str, adapted_modules: str, llmConfig: LLMConfig = LLMConfig(), print_in_out:bool = False) -> dict:
    
    reasoning_structure = None
    
    if llmConfig.model_type == ModelType.OPENAI:
        reasoning_structure = __implement_openai(llmConfig=llmConfig,task=task,adapted_modules=adapted_modules, print_in_out=print_in_out)
    else: 
        raise ValueError("Unsupported model Type!")
    
    return reasoning_structure

def __implement_openai(llmConfig: LLMConfig, task: str, adapted_modules: str, print_in_out:bool = False):
    prompt = config.IMPLEMENT_PHASE_PROMPT_TEMPLATE.format(task=task,modules=adapted_modules)
    response = openai.invoke(prompt,api_key=llmConfig.api_key,temp=llmConfig.temp,max_context=llmConfig.context_length,model_name=llmConfig.model_name,print_in_out=print_in_out)
    reasoning_structure = utils.extractJSONToDict(response=response)
    return reasoning_structure
    


def self_discover(task: str, llmConfig: LLMConfig = LLMConfig(),verbose=False,retries = 3, print_in_out:bool = False) -> dict:
    
    # SELECT
    selection = None 
    numAttempts = 0
    log_print("Starting SELECT Phase")
    while selection is None and numAttempts < retries:
        try:
            selection = select(task= task,llmConfig=llmConfig, print_in_out=print_in_out)
        except Exception as e:
            numAttempts += 1
            log_print(f"Failed to select Reasoning Modules due to {e}...")
    if selection is None: raise Exception("Unable to Reasoning Module selection from LLM response.")
    log_print("SELECT Phase Complete")
    if verbose: 
        
        module_list = ""
        for module in selection["reasoning_modules"]:
            module_list += f"- {module}. {utils.id_to_rm(module)}\n"
        log_print(f"Reasoning Modules Picked:\n{module_list}")
            
    # ADAPT
    adapted_modules = None
    numAttempts = 0
    log_print("Starting ADAPT Phase")
    while adapted_modules is None and numAttempts < retries:
        try:
            adapted_modules = adapt(task=task,reasoning_modules=selection,llmConfig=llmConfig, print_in_out=print_in_out)
        except Exception as e:
            numAttempts += 1
            log_print(f"Failed to rephrase Reasoning Modules due to {e}...")
    if adapted_modules is None: raise Exception("Unable to extract adapted_modules from LLM response.")
    log_print("ADAPT Phase Complete")
    if verbose: 
        log_print(f"Task-specific Reasoning Module verbiage:\n{adapted_modules}")
    
    # IMPLEMENT
    reasoning_structure = None
    numAttempts = 0
    log_print("Starting IMPLEMENT Phase")
    while reasoning_structure is None and numAttempts < retries:
        try:
            reasoning_structure = implement(task=task,adapted_modules=adapted_modules,llmConfig=llmConfig, print_in_out=print_in_out)
        except:
            numAttempts += 1
            if verbose: log_print(f"Failed to construct Reasoning Structure in JSON. Starting attempt {numAttempts+1}/{retries} ...")
    if reasoning_structure is None: raise Exception("Unable to extract reasoning structure from LLM response.")
    log_print("IMPLEMENT Phase Complete")
    if verbose: 
        log_print(f"Reasoning Structure:\n{json.dumps(reasoning_structure, indent=2)}")
        
    return reasoning_structure
    
def solve(task: str, discover_config: LLMConfig = LLMConfig(), solve_config: LLMConfig = None,verbose=False,retries=3, print_in_out:bool = False) -> str:
    
    if verbose: log_print(f"discover_config: {discover_config}\nsolve_config: {solve_config}")
    reasoning_structure = self_discover(task=task,llmConfig=discover_config,verbose=verbose,retries=retries, print_in_out=print_in_out)
    
    if not solve_config:
        solve_config = discover_config
    if solve_config.model_type == ModelType.OPENAI:
        answer = __solve_openai(task=task,llmConfig=solve_config,reasoning_structure=reasoning_structure,verbose=verbose,retries=retries, print_in_out=print_in_out)
    else: 
        raise ValueError("Unsupported model Type!")
    log_print("Solution has been found.")
    return answer
    

def __solve_openai(task: str, llmConfig: LLMConfig,reasoning_structure: dict,verbose=False,retries=3, print_in_out:bool = False) -> str:
    
    reasoning_structure_str = json.dumps(reasoning_structure,indent=2)
    prompt = config.SOLVE_PROMPT_TEMPLATE.format(task=task,reasoning_structure=reasoning_structure_str)
    numAttempts = 0
    answer = None
    log_print("Starting to Solve Problem using Reasoning Structure")
    while answer is None and numAttempts < retries:
        try:
            response = openai.invoke(prompt,api_key=llmConfig.api_key,temp=llmConfig.temp,max_context=llmConfig.context_length,model_name=llmConfig.model_name,print_in_out=print_in_out)
            reasoning = utils.extractJSONToDict(response)
            answer = None
            try:
                answer = reasoning["Reasoning Structure"]["FINAL_ANSWER"]
            except:
                pass
            try:
                answer = reasoning["FINAL_ANSWER"]
            except:
                pass
            
            if not answer:
                raise ValueError("Unable to exxtract FINAL_ANSWER from completed reasoning structure")
        except Exception as e:
            numAttempts += 1
            if verbose: log_print(f"Failed to Extract answer. Exception: {e} . Starting attempt {numAttempts+1}/{retries} ...")
    if verbose: log_print(f"Problem Solved\nCompleted Reasoning Structure:\n{json.dumps(reasoning,indent=2)}")
    return answer

def __select_openai(llmConfig: LLMConfig, task: str, print_in_out:bool = False) -> dict:
    
    prompt = config.SELECT_PHASE_PROMPT_TEMPLATE.format(task=task,modules=utils.rm_list())
    response = openai.invoke(prompt,api_key=llmConfig.api_key,temp=llmConfig.temp,max_context=llmConfig.context_length,model_name = llmConfig.model_name,print_in_out=print_in_out)
    selection = utils.extractJSONToDict(response=response)
    return selection

def __adapt_openai(llmConfig: LLMConfig, task: str, reasoning_modules: dict, print_in_out:bool = False) -> str:
    
    module_list = ""
    for module in reasoning_modules["reasoning_modules"]:
        module_list += f"- {module}. {utils.id_to_rm(module)}\n"
        
    prompt = config.ADAPT_PHASE_PROMPT_TEMPLATE.format(task=task,modules=module_list)
    response = openai.invoke(prompt,api_key=llmConfig.api_key,temp=llmConfig.temp,max_context=llmConfig.context_length,model_name=llmConfig.model_name,print_in_out=print_in_out)
    adapted_modules = utils.extractMDBlock(response=response)
    return adapted_modules
