# AgentSmith

A helper library for building, testing, and monitoring LangGraph workflows with minimal boilerplate.

## Overview

AgentSmith provides a comprehensive toolkit for creating production-ready GenAI applications using LangGraph and MLflow:

- **🔧 Model Management** - Unified interface for multiple LLM providers with YAML configuration
- **📝 Prompt Management** - Versioned prompts with hot-reloading and structured response parsing
- **🏗️ Graph Building** - Simplified API for creating complex LangGraph workflows  
- **🧪 Evaluation Framework** - MLflow integration for testing and monitoring
- **📊 Node Framework** - Minimal boilerplate for creating LLM-powered nodes
- **🚀 App Versioning** - Automatic MLflow LoggedModel creation with trace linking for GenAI apps

## Quick Start

```python
from agentsmith import get_model, create_graph
from agentsmith.run import run_evaluation

# 1. Get a configured model
llm = get_model("default")  # Loads from models.yaml

# 2. Create a simple workflow
def chat_node(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

app = create_graph(MyState) \
    .add_nodes({"chat": chat_node}) \
    .connect_nodes("START", "chat") \
    .connect_nodes("chat", "END") \
    .compile()

# 3. Run evaluation
run_evaluation(
    app=app,
    data="test_data.csv",
    scorers=[correctness_scorer]
)
```

## Core Modules

### 🔧 Model Management (`agentsmith.config.models`)

Centralized configuration for multiple LLM providers with hot-reloading support.

```python
from agentsmith import get_model, register_model, ModelConfig

# Load from YAML configuration
llm = get_model("claude-3.7-sonnet")

# Or register programmatically  
register_model(ModelConfig(
    name="custom-model",
    provider="bedrock",
    model_id="arn:aws:bedrock:...",
    max_tokens=4096
))

# List available models
print(list_available_models())
```

**Configuration Example** (`agentsmith/config/models.yaml`):
```yaml
default_model: claude-3.7-sonnet

models:
  claude-3.7-sonnet:
    provider: bedrock
    model_id: ${CLAUDE_37_SONNET_ARN}
    max_tokens: 8192
    temperature: 0.0
    
  gpt-4:
    provider: openai
    model_id: gpt-4
    api_key_env: OPENAI_API_KEY
    max_tokens: 4096
```

### 📝 Prompt Management (`agentsmith.config.prompts`)

Version-controlled prompts with structured response parsing and hot-reloading.

```python
from agentsmith.config.prompts import PromptManager

prompt_manager = PromptManager("prompts.yaml")

# Get versioned prompt
template, schema = prompt_manager.get_prompt("classify_author", version="1.2")

# Format with variables
formatted = template.format(author_name="Stephen King", context="...")

# Parse structured response
parsed = prompt_manager.parse_response(llm_output, schema)
print(parsed)  # {"category": "Fiction", "certainty": 0.9}
```

**Configuration Example** (`agentsmith/config/prompts.yaml`):
```yaml
prompts:
  classify_author:
    description: "Classify authors into categories"
    default_version: "1.2"
    versions:
      "1.2":
        template: |
          Classify this author: {author_name}
          Context: {context}
        response:
          format: labeled
          fields:
            category:
              type: enum
              choices: ["Fiction", "Non-Fiction", "Poetry"]
            certainty:
              type: float
              range: [0.0, 1.0]
```

### 🏗️ Graph Building (`agentsmith.graph.builder`)

Simplified API for creating LangGraph workflows with intuitive node connections.

```python
from agentsmith import create_graph
from langgraph.graph import START, END

# Define state
class WorkflowState(TypedDict):
    input: str
    result: str
    messages: List[BaseMessage]

# Create nodes
def process_node(state):
    return {"result": f"Processed: {state['input']}"}

def validate_node(state):
    return {"messages": [AIMessage("Validation complete")]}

# Build workflow
app = create_graph(WorkflowState) \
    .add_nodes({
        "process": process_node,
        "validate": validate_node
    }) \
    .connect_nodes(START, "process") \
    .connect_nodes("process", "validate") \
    .connect_nodes("validate", END) \
    .compile()

# Run workflow
result = app.invoke({"input": "Hello world"})
```

**Advanced Routing:**
```python
def should_continue(state):
    return "validate" if state.get("needs_validation") else "END"

app = create_graph(MyState) \
    .add_nodes({"process": process_node, "validate": validate_node}) \
    .connect_nodes(START, "process") \
    .connect_nodes("process", ["validate", "END"], condition=should_continue) \
    .compile()
```

### 🧪 Node Framework (`agentsmith.graph.nodes`)

Minimal boilerplate for creating LLM-powered nodes with automatic prompt integration.

```python
from agentsmith.graph.nodes import NodeConfig, create_llm_node

# Configure LLM node
config = NodeConfig(
    name="author_classifier",
    prompt_name="classify_author",
    input_fields=["author_name", "context"],
    include_messages=True
)

# Create node function
classifier_node = create_llm_node(config, llm, prompt_manager)

# Use in workflow
app = create_graph(MyState) \
    .add_nodes({"classify": classifier_node}) \
    .connect_nodes(START, "classify") \
    .connect_nodes("classify", END) \
    .compile()
```

### 📊 Evaluation Framework (`agentsmith.run`)

Streamlined MLflow integration for testing and monitoring LangGraph applications.

```python
from agentsmith.run import run_evaluation, prepare_eval_data
from agentsmith.run.scorers import equality_scorer, grounded_scorer

# Prepare evaluation data
eval_data = prepare_eval_data(
    "test_authors.csv",
    input_columns=["author_name"],
    expectation_mapping={"correct_tag": "expected_response"}
)

# Create scorers
tag_accuracy = equality_scorer(
    name="tag_accuracy", 
    output_field="result.tag",
    expectation_field="expected_response"
)

bio_quality = grounded_scorer(
    name="bio_grounded",
    output_field="state.bio",
    facts_field="expected_facts"
)

# Run evaluation
results = run_evaluation(
    app=my_app,
    data=eval_data,
    scorers=[tag_accuracy, bio_quality],
    experiment_name="author_tagging_v2"
)

print(f"Tag Accuracy: {results.average_scores['tag_accuracy']}")
```

**Custom Scorers:**
```python
from agentsmith.run.scorers import guideline_scorer

quality_scorer = guideline_scorer(
    name="reasoning_quality",
    guidelines="""
    Evaluate if the reasoning is logical and well-supported.
    Score 'yes' if arguments are coherent and evidence-based.
    """,
    response_field="result.notes"
)
```

### 🚀 App Runner (`agentsmith.run.runners`)

Standardized execution with MLflow tracing and automatic GenAI app versioning for consistent monitoring.

```python
from agentsmith.run.runners import run_app, create_runner

# Simple execution - auto-generates model version "{llm}-v1"
result = run_app(
    app=my_app,
    inputs={"author_name": "Stephen King"},
    result_field="chosen_tag_triple"
)
# Creates MLflow model: "claude-3-7-sonnet-v1" with LLM parameter logged

# Explicit versioning with app metadata
result = run_app(
    app=my_app,
    inputs={"author_name": "Stephen King"},
    result_field="chosen_tag_triple",
    metadata={
        "prompt_template": "v2.1", 
        "search_provider": "serpapi",
        "temperature": "0.2"
    },
    version_name="author-tagger-v2"
)
# Creates MLflow model: "author-tagger-v2" with LLM + all metadata logged

# Create reusable runner with versioning
run_my_app = create_runner(
    app=my_app,
    result_field="result",
    metadata={"prompt_version": "3.0"},
    version_name="my-app-v3"
)

# All traces automatically linked to the versioned model
result1 = run_my_app({"query": "What is AI?"})
result2 = run_my_app({"query": "Explain LangGraph"})
```

**MLflow LoggedModel Integration:**
- **Automatic Versioning**: Creates MLflow LoggedModel for each GenAI app version
- **Trace Linking**: All execution traces automatically associated with the model version
- **Parameter Logging**: Logs LLM and app-level parameters for version tracking
- **Databricks UI**: View model versions, parameters, and linked traces in Databricks
- **Name Sanitization**: Automatically handles invalid characters in model names

## Configuration Files

### Environment Setup
```bash
# Required environment variables
export CLAUDE_37_SONNET_ARN="arn:aws:bedrock:us-east-1:..."
export OPENAI_API_KEY="sk-..."
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### Directory Structure
```
your-project/
├── agentsmith/
│   └── config/
│       ├── models.yaml      # Model configurations
│       └── prompts.yaml     # Prompt templates
├── your_workflow.py         # Your LangGraph application
└── evaluation.py           # Evaluation scripts
```

## Best Practices

### 1. **Model Management**
```python
# Use environment-specific configurations
llm = get_model("default")  # Production model
llm_fast = get_model("claude-haiku")  # Development model

# Cache models globally
from agentsmith import get_model_manager
manager = get_model_manager()
```

### 2. **Prompt Versioning**
```python
# Use semantic versioning for prompts
template, schema = prompt_manager.get_prompt("classify", version="1.2")

# Enable auto-versioning for experiments
template, schema = prompt_manager.get_prompt("classify", use_latest=True)
```

### 3. **Evaluation Strategy**
```python
# Create comprehensive evaluation suites
scorers = [
    equality_scorer("accuracy", "result.tag", "expected_tag"),
    grounded_scorer("factual", "state.bio", "expected_facts"),
    guideline_scorer("quality", "Response should be clear and concise")
]

# Run evaluations in experiments
run_evaluation(
    app=app,
    data=eval_data,
    scorers=scorers,
    experiment_name="my_experiment_v1",
    run_name=f"run_{datetime.now().isoformat()}"
)
```

### 4. **GenAI App Versioning**
```python
# Development: Use auto-generated versions for quick iteration
result = run_app(app, inputs)  # Creates "{llm}-v1" model

# Production: Use explicit versions with meaningful metadata
result = run_app(
    app=app,
    inputs=inputs,
    metadata={
        "prompt_template": "classification_v2.1",
        "model_config": "production",
        "search_provider": "serpapi_enhanced"
    },
    version_name="author-classifier-prod-v2-1"
)

# Create versioned runners for consistent app deployment
production_runner = create_runner(
    app=app,
    metadata={"env": "prod", "version": "2.1"},
    version_name="my-app-prod-v2-1"
)
```

### 5. **Error Handling**
```python
# Apps automatically handle errors in standardized format
result = run_app(app, inputs)
if "error" in result["result"]:
    print(f"Error: {result['result']['error']}")
else:
    print(f"Success: {result['result']}")
```

## Integration Examples

### Complete Workflow Example
```python
from agentsmith import get_model, create_graph, PromptManager
from agentsmith.run import run_evaluation, run_app
from agentsmith.run.scorers import equality_scorer

# 1. Setup
llm = get_model("default")
prompt_manager = PromptManager("config/prompts.yaml")

# 2. Define state
class AuthorState(TypedDict):
    author_name: str
    classification: str
    confidence: float

# 3. Create workflow
def classify_node(state):
    template, schema = prompt_manager.get_prompt("classify_author")
    prompt = template.format(author_name=state["author_name"])
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = prompt_manager.parse_response(response.content, schema)
    return {"classification": parsed["category"], "confidence": parsed["certainty"]}

app = create_graph(AuthorState) \
    .add_nodes({"classify": classify_node}) \
    .connect_nodes("START", "classify") \
    .connect_nodes("classify", "END") \
    .compile()

# 4. Test individual execution with versioning
result = run_app(
    app=app,
    inputs={"author_name": "Stephen King"},
    result_field="classification",
    metadata={
        "prompt_version": "classify_author_v1.2",
        "llm_temperature": "0.0"
    },
    version_name="author-classifier-v1-2"
)
print(f"Classification: {result['result']}")

# 5. Run full evaluation (automatically links traces to the versioned model)
results = run_evaluation(
    app=app,
    data="authors.csv",
    scorers=[equality_scorer("accuracy", "classification", "expected_tag")]
)
print(f"Accuracy: {results.metrics['accuracy/mean']}")
```

## Migration Guide

### From Raw LangGraph
```python
# Before: Raw LangGraph
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(MyState)
workflow.add_node("node1", func1)
workflow.add_node("node2", func2)
workflow.add_edge(START, "node1")
workflow.add_edge("node1", "node2")
workflow.add_edge("node2", END)
app = workflow.compile()

# After: AgentSmith
from agentsmith import create_graph

app = create_graph(MyState) \
    .add_nodes({"node1": func1, "node2": func2}) \
    .connect_nodes("START", "node1") \
    .connect_nodes("node1", "node2") \
    .connect_nodes("node2", "END") \
    .compile()
```

### From Manual MLflow
```python
# Before: Manual MLflow setup
import mlflow
with mlflow.start_run():
    results = []
    for data in eval_data:
        result = app.invoke(data["inputs"])
        results.append(result)
    # Manual scoring logic...

# After: AgentSmith
from agentsmith.run import run_evaluation
results = run_evaluation(app=app, data=eval_data, scorers=scorers)
```

---

AgentSmith transforms complex LangGraph and MLflow workflows into simple, maintainable code that scales from prototype to production.