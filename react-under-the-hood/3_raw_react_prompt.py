import re
import os
import inspect
import json
from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI
from typing import List

load_dotenv()


MAX_AGENT_ITERATIONS = 10
MODEL = "gpt-oss-120b"
llm_client = OpenAI(
    base_url="https://backend.sovereigneg.com/v1",
    api_key=os.getenv('SOVEREIGNEG_API_KEY')
)

# --- Tools ---
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up a price of a product in the catalog"""
    print(f"    >> Executing get_product_price({product})")
    prices = {"Asus Tuf F15": 1299.00, "Dell G15": 1199.99}
    print(f"[GET_PRODUCT_PRICE]: product {product}")
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers: bronze, sliver, gold."""
    discount_tiers = {"bronze": 0.01, "silver": 0.05, "gold": 0.1}
    price = float(price)
    return round(price - discount_tiers.get(discount_tier, 0) * price, 2)


tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}


def get_tools_description(tools_dict) -> str:
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tools_description = get_tools_description(tools)
tools_names = ", ".join(tools.keys())


ReAct_prompt = """
STRICT RULES: you must follow these exactly:
1. Do not ever assume any product price. 
2. You must call get_product_price first to get the real price.
3. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price and don't made up a price.
4. Don't calculate discounts yourself, always use apply_discount tool.
5. If a user doesn't specify a discount tier ASK them first and do not assume one.

Answer the following questions as best you can. You have access to the following tools:

{tools_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_names}]
Action Input: a JSON object containing the tool arguments
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:"""



@traceable(name="OpenAI", run_type="llm")
def sovereigneg_ai(model, messages, options):
    return llm_client.chat.completions.create(
        model=model,
        messages=messages,
        stop=options.get("stop"),
        temperature=options.get("temperature", 0)
    )


# --- Agent Loop ---
@traceable(name="Raw ReAct Prompt Agent Loop")
def run_agent(question: str):

    prompt = ReAct_prompt.format(
        tools_description=tools_description,
        tools_names=tools_names,
        input=question
    )
    scratchpad = ""

    for i in range(1, MAX_AGENT_ITERATIONS + 1):
        print(f"[Iteration]: {i}")
        full_prompt = prompt + scratchpad
        response = sovereigneg_ai(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation:"], "temperature": 0}
        )
        output = response.choices[0].message.content
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        if not action_input_match or not action_match:
            print(
                f"[Parsing] Error: could not parse Action/Action Input from LLM response."
            )
            return None

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
         
        print(f"[Tools]: tool {tool_name} selected with args {tool_input_raw}")

        try:
            kwargs = json.loads(tool_input_raw)
        except json.JSONDecodeError as e:
            observation = f"[ERROR]: Invalid JSON Action Input: {e}"
        else:
            if tool_name not in tools:
                observation = (
                    f"[ERROR]: Tool name {tool_name} not found "
                    f"in {list(tools.keys())}"
                )
            else:
                try:
                    observation = str(tools[tool_name](**kwargs))
                except Exception as e:
                    observation = f"[ERROR]: Tool execution failed: {e}"

        print(f"[ToolsObservation]: {tool_name}({kwargs}) -> {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print(f"Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    result = run_agent("What is the price of 'Asus Tuf F15' after apply gold discount?")
    print(result)
