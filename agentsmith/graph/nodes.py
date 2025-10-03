"""
Node framework for creating LangGraph nodes with minimal boilerplate.
"""
from typing import Dict, Any, Callable, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage

if TYPE_CHECKING:
    from ..config.prompts import PromptManager


@dataclass
class NodeConfig:
    """Configuration for an LLM-based LangGraph node."""
    name: str
    prompt_name: str
    input_fields: List[str] = field(default_factory=list)
    pre_processor: Optional[Callable[[Dict], Optional[Dict]]] = None
    post_processor: Optional[Callable[[Dict, Dict], Dict]] = None
    include_messages: bool = True
    message_template: str = "{name} completed"


def create_llm_node(
    config: NodeConfig,
    llm,
    prompt_manager: 'PromptManager'
) -> Callable[[Dict], Dict]:
    """
    Create an LLM-based LangGraph node.
    
    Args:
        config: Node configuration
        llm: Language model to use
        prompt_manager: PromptManager instance for loading prompts
        
    Returns:
        A callable node function for LangGraph
    """
    
    def node_function(state: Dict[str, Any]) -> Dict:
        # Pre-processing (optional early exit)
        if config.pre_processor:
            pre_result = config.pre_processor(state)
            if pre_result is not None:
                return pre_result
        
        # Extract variables from state
        variables = {
            field: state.get(field, "") 
            for field in config.input_fields
        }
        
        # Get prompt and invoke LLM
        prompt_template, response_schema = prompt_manager.get_prompt(config.prompt_name)
        prompt = HumanMessage(content=prompt_template.format(**variables)) # Populate prompt template with variables. HumanMessage is langchain wrapper for user prompts. 
        raw_response = llm.invoke([prompt])
        
        # Parse response
        parsed_response = prompt_manager.parse_response(raw_response.content, response_schema)
        
        # Post-process if needed
        if config.post_processor:
            output = config.post_processor(parsed_response, state)
        else:
            output = parsed_response

        # Add standard message if needed
        if config.include_messages and "messages" not in output:
            output["messages"] = [AIMessage(content=config.message_template.format(name=config.name))] # BUG this file doesn't exist
        
        return output
    
    node_function.__name__ = config.name
    return node_function


def create_custom_node(name: str, func: Callable) -> Callable:
    """
    Wrap a custom function as a named node.
    
    Args:
        name: Name for the node
        func: The function to wrap
        
    Returns:
        The same function with __name__ set
    """
    func.__name__ = name
    return func