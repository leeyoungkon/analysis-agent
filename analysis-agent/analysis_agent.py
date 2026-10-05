# analysis_agent.py

import os
import json
import asyncio

import requests
import pymysql
import pandas as pd

from mcp_client import CustomerMCPClient


# =========================================================
# 환경 설정
# =========================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama:11434/api/chat"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:12b"
)

ANALYTICS_DB_HOST = os.getenv(
    "ANALYTICS_DB_HOST",
    "mysql"
)

ANALYTICS_DB_PORT = int(
    os.getenv(
        "ANALYTICS_DB_PORT",
        "3306"
    )
)

ANALYTICS_DB_USER = os.getenv(
    "ANALYTICS_DB_USER",
    "root"
)

ANALYTICS_DB_PASSWORD = os.getenv(
    "ANALYTICS_DB_PASSWORD",
    "root1234"
)

ANALYTICS_DB_NAME = os.getenv(
    "ANALYTICS_DB_NAME",
    "analytics"
)

MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "http://customer-mcp:8000/mcp"
)


mcp_client = CustomerMCPClient(
    MCP_SERVER_URL
)


# =========================================================
# Analytics DB
# =========================================================

def get_analytics_connection():

    return pymysql.connect(
        host=ANALYTICS_DB_HOST,
        port=ANALYTICS_DB_PORT,
        user=ANALYTICS_DB_USER,
        password=ANALYTICS_DB_PASSWORD,
        database=ANALYTICS_DB_NAME,
        cursorclass=pymysql.cursors.DictCursor
    )


# =========================================================
# Analytics Tool 1
# 고객별 주문량 추이
# =========================================================

def get_customer_trend(
    customer_id: str | None = None
) -> pd.DataFrame:

    conn = get_analytics_connection()

    try:

        if customer_id:

            sql = """
            SELECT
                minute_time,
                customer_id,
                total_quantity,
                total_amount,
                order_count
            FROM customer_order_minute
            WHERE customer_id = %s
            ORDER BY minute_time
            """

            df = pd.read_sql(
                sql,
                conn,
                params=(customer_id,)
            )

        else:

            sql = """
            SELECT
                minute_time,
                customer_id,
                total_quantity,
                total_amount,
                order_count
            FROM customer_order_minute
            ORDER BY minute_time, customer_id
            """

            df = pd.read_sql(
                sql,
                conn
            )

        return df

    finally:
        conn.close()


# =========================================================
# Analytics Tool 2
# 고객별 총 주문량 / 매출
# =========================================================

def get_customer_sales() -> pd.DataFrame:

    conn = get_analytics_connection()

    sql = """
    SELECT
        customer_id,
        SUM(total_quantity) AS total_quantity,
        SUM(total_amount) AS total_sales,
        SUM(order_count) AS total_orders
    FROM customer_order_minute
    GROUP BY customer_id
    ORDER BY total_sales DESC
    """

    try:

        df = pd.read_sql(
            sql,
            conn
        )

        return df

    finally:
        conn.close()


# =========================================================
# Analytics Tool 3
# 고객별 증가량 분석
# =========================================================

def get_customer_growth() -> pd.DataFrame:

    conn = get_analytics_connection()

    sql = """
    WITH ranked AS (

        SELECT
            customer_id,
            minute_time,
            total_quantity,

            ROW_NUMBER() OVER (
                PARTITION BY customer_id
                ORDER BY minute_time ASC
            ) AS rn_first,

            ROW_NUMBER() OVER (
                PARTITION BY customer_id
                ORDER BY minute_time DESC
            ) AS rn_last

        FROM customer_order_minute
    ),

    first_last AS (

        SELECT
            customer_id,

            MAX(
                CASE
                    WHEN rn_first = 1
                    THEN total_quantity
                END
            ) AS first_quantity,

            MAX(
                CASE
                    WHEN rn_last = 1
                    THEN total_quantity
                END
            ) AS last_quantity

        FROM ranked

        GROUP BY customer_id
    )

    SELECT
        customer_id,
        first_quantity,
        last_quantity,

        last_quantity - first_quantity
            AS quantity_growth,

        CASE

            WHEN first_quantity = 0
            THEN NULL

            ELSE
                ROUND(
                    (
                        (
                            last_quantity
                            - first_quantity
                        )
                        / first_quantity
                    ) * 100,
                    2
                )

        END AS growth_rate

    FROM first_last

    ORDER BY quantity_growth DESC
    """

    try:

        df = pd.read_sql(
            sql,
            conn
        )

        return df

    finally:
        conn.close()


# =========================================================
# DataFrame → JSON
# =========================================================

def dataframe_to_records(
    df: pd.DataFrame,
    limit: int = 100
) -> list:

    if df.empty:
        return []

    result = df.head(limit).copy()

    # datetime JSON 변환
    for col in result.columns:

        if pd.api.types.is_datetime64_any_dtype(
            result[col]
        ):
            result[col] = (
                result[col]
                .astype(str)
            )

    result = result.where(
        pd.notnull(result),
        None
    )

    return result.to_dict(
        orient="records"
    )


# =========================================================
# Ollama 호출
# =========================================================

def call_llm(
    messages: list,
    json_mode: bool = False
) -> str:

    payload = {

        "model": OLLAMA_MODEL,

        "messages": messages,

        "stream": False
    }

    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data[
        "message"
    ][
        "content"
    ]


# =========================================================
# LLM Planner
# =========================================================

def create_plan(
    question: str
) -> dict:

    system_prompt = """
당신은 기업 고객 주문 분석 AI Agent이다.

사용자의 자연어 질문을 분석하여
필요한 Tool 실행 계획을 JSON으로 작성한다.

사용 가능한 Tool은 다음과 같다.

1. trend
고객별 주문량 시계열 추이를 분석한다.

2. sales
고객별 총 주문량, 총 주문금액,
총 주문건수를 분석한다.

3. growth
고객별 주문량 증가 정도를 분석한다.

4. customer_info
customer_id로 고객 기본정보를 조회한다.
이 Tool은 MCP Server를 사용한다.

5. search_customer
고객명으로 고객을 검색한다.
이 Tool은 MCP Server를 사용한다.


여러 Tool을 순차적으로 사용할 수도 있다.

예:

질문:
"고객 C001의 주문 추이와 고객 정보를 알려줘"

결과:

{
  "actions": [
    {
      "tool": "trend",
      "customer_id": "C001"
    },
    {
      "tool": "customer_info",
      "customer_id": "C001"
    }
  ]
}


질문:
"최근 주문량이 가장 증가한 고객의 정보를 알려줘"

결과:

{
  "actions": [
    {
      "tool": "growth"
    },
    {
      "tool": "customer_info",
      "customer_id": "$top_customer_id"
    }
  ]
}


규칙:

- 반드시 JSON만 출력한다.
- 존재하지 않는 customer_id를 임의로 만들지 않는다.
- 질문에 고객ID가 없으면 null 또는 생략한다.
- 분석 결과를 이용해야 하는 경우
  $top_customer_id 같은 변수를 사용할 수 있다.
"""

    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": question
        }
    ]

    result = call_llm(
        messages,
        json_mode=True
    )

    try:

        return json.loads(result)

    except Exception:

        return {
            "actions": []
        }


# =========================================================
# Plan 실행
# =========================================================

async def execute_plan(
    plan: dict
) -> dict:

    results = {}

    context = {}

    actions = plan.get(
        "actions",
        []
    )

    for action in actions:

        tool = action.get(
            "tool"
        )

        # ---------------------------------------------
        # trend
        # ---------------------------------------------

        if tool == "trend":

            customer_id = action.get(
                "customer_id"
            )

            if customer_id == "$top_customer_id":

                customer_id = context.get(
                    "top_customer_id"
                )

            df = get_customer_trend(
                customer_id
            )

            results["trend"] = (
                dataframe_to_records(
                    df
                )
            )


        # ---------------------------------------------
        # sales
        # ---------------------------------------------

        elif tool == "sales":

            df = get_customer_sales()

            records = dataframe_to_records(
                df
            )

            results["sales"] = records

            if records:

                context[
                    "top_customer_id"
                ] = records[0][
                    "customer_id"
                ]


        # ---------------------------------------------
        # growth
        # ---------------------------------------------

        elif tool == "growth":

            df = get_customer_growth()

            records = dataframe_to_records(
                df
            )

            results["growth"] = records

            if records:

                context[
                    "top_customer_id"
                ] = records[0][
                    "customer_id"
                ]


        # ---------------------------------------------
        # customer_info
        # MCP 호출
        # ---------------------------------------------

        elif tool == "customer_info":

            customer_id = action.get(
                "customer_id"
            )

            if customer_id == "$top_customer_id":

                customer_id = context.get(
                    "top_customer_id"
                )

            if customer_id:

                customer = (
                    await mcp_client
                    .get_customer_info(
                        customer_id
                    )
                )

                results[
                    "customer_info"
                ] = customer


        # ---------------------------------------------
        # search_customer
        # ---------------------------------------------

        elif tool == "search_customer":

            customer_name = action.get(
                "customer_name"
            )

            if customer_name:

                customer = (
                    await mcp_client
                    .search_customer(
                        customer_name
                    )
                )

                results[
                    "customer_search"
                ] = customer

    return results


# =========================================================
# 최종 LLM 응답
# =========================================================

def generate_final_answer(
    question: str,
    tool_results: dict
) -> str:

    system_prompt = """
당신은 고객 주문 데이터를 분석하는
기업용 AI 분석가이다.

사용자 질문과 Tool 실행 결과를 바탕으로
정확하고 이해하기 쉬운 한국어 답변을 작성한다.

규칙:

1. 제공되지 않은 데이터는 추측하지 않는다.
2. 숫자를 임의로 생성하지 않는다.
3. 고객 정보는 MCP 결과만 사용한다.
4. 분석 결과가 없으면
   데이터가 없다고 명확히 말한다.
5. 중요한 수치와 고객을 먼저 설명한다.
6. 필요하면 간단한 분석 해석을 덧붙인다.
"""

    result_text = json.dumps(
        tool_results,
        ensure_ascii=False,
        indent=2,
        default=str
    )

    user_prompt = f"""
사용자 질문:

{question}


Tool 실행 결과:

{result_text}


위 결과를 바탕으로 최종 답변을 작성하라.
"""

    return call_llm(
        [

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )


# =========================================================
# Analysis Agent
# =========================================================

async def analyze_agent(
    question: str
) -> dict:

    print(
        "\n=============================="
    )

    print(
        " Analysis Agent"
    )

    print(
        "=============================="
    )

    print(
        f"[User] {question}"
    )


    # ---------------------------------------------
    # 1. LLM Planning
    # ---------------------------------------------

    plan = create_plan(
        question
    )

    print(
        "[Agent Plan]"
    )

    print(
        json.dumps(
            plan,
            ensure_ascii=False,
            indent=2
        )
    )


    # ---------------------------------------------
    # 2. Tool 실행
    # ---------------------------------------------

    tool_results = await execute_plan(
        plan
    )

    print(
        "[Tool Results]"
    )

    print(
        json.dumps(
            tool_results,
            ensure_ascii=False,
            indent=2,
            default=str
        )
    )


    # ---------------------------------------------
    # 3. LLM Final Answer
    # ---------------------------------------------

    answer = generate_final_answer(
        question,
        tool_results
    )

    return {

        "question": question,

        "plan": plan,

        "tool_results": tool_results,

        "answer": answer
    }


# =========================================================
# CLI 실행
# =========================================================

async def main():

    print(
        "=============================="
    )

    print(
        " LLM + MCP Analysis Agent"
    )

    print(
        "=============================="
    )

    while True:

        question = input(
            "\n분석 질문 입력(q=종료): "
        ).strip()

        if question.lower() in [
            "q",
            "quit",
            "exit"
        ]:
            break

        try:

            result = await analyze_agent(
                question
            )

            print(
                "\n=============================="
            )

            print(
                " Answer"
            )

            print(
                "=============================="
            )

            print(
                result["answer"]
            )

        except Exception as e:

            print(
                f"[ERROR] {e}"
            )


if __name__ == "__main__":

    asyncio.run(
        main()
    )