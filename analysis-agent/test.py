import asyncio
from analysis_agent import analyze_agent

result = asyncio.run(
    analyze_agent(
        "최근 주문량이 가장 증가한 고객의 이름과 등급을 알려줘"
    )
)
print(result["plan"])
print(result["tool_results"])
print(result["answer"])
import inspect
print(inspect.iscoroutinefunction(analyze_agent))
