from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()


MAX_AGENT_ITERATIONS = 10
MODEL = "gemini-2.5-flash-lite"


# --- Tools ---
@tool
def get_product_price(product: str) -> float:
    """Look up a price of a product in the catalog"""
    print(f"    >> Executing get_product_price({product})")
    prices = {"Asus Tuf F15": 1299.00, "Dell G15": 1199.99}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers: bronze, sliver, gold."""
    discount_tiers = {"bronze": 0.01, "silver": 0.05, "gold": 0.1}
    return round(price - discount_tiers.get(discount_tier, 0) * price, 2)


# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(
        MODEL,
        model_provider="google_genai",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to the following tools:\n"
                "1. product price catalog tool\n"
                "2. apply discount tool with discount tiers\n"
                "STRICT RULES: you must follow these exactly:\n"
                "1. Do not ever assume any product price.\n"
                "2. You must call get_product_price first to get the real price.\n"
                "3. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price and don't made up a price.\n"
                "4. Don't calculate discounts yourself, always use apply_discount tool.\n"
                "5. If a user doesn't specify a discount tier ASK them first and do not assume one.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for i in range(1, MAX_AGENT_ITERATIONS + 1):
        print(f"[Iteration]: {i}")
        response = llm_with_tools.invoke(messages)
        if not response.tool_calls:
            return response.content
        tool_call = response.tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"[Tools]: tool {tool_name} selected with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        observation = tool_to_use.invoke(tool_args)

        print(f"[ToolsObservation]: {observation}")

        messages.append(response)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print(f"Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    result = run_agent("What is the price of 'Asus Tuf F15' after apply gold discount?")
    print(result)
