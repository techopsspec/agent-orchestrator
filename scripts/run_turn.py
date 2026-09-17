"""Manual driver for one deep-agent turn -- for iterating on the agent wiring directly, without
the FastAPI layer. Requires an identity token minted by Rails, e.g.:

    cd Operator-Portal/backend && RAILS_ENV=development bin/rails runner \\
      'puts McpTools::IdentityToken.mint(user: User.find(1), conversation: ChatConversation.first)'

Usage: IDENTITY_TOKEN=... python scripts/run_turn.py "What snowfall zones exist?"
"""

import asyncio
import os
import sys

from langchain_core.messages import HumanMessage

from agent_orchestrator.apps.operator_portal import MAX_TOOL_ROUNDTRIPS, build_agent


async def main():
    identity_token = os.environ["IDENTITY_TOKEN"]
    message = sys.argv[1] if len(sys.argv) > 1 else "What snowfall zones exist?"

    agent, provider_map = await build_agent(identity_token)
    print(f"provider map: {provider_map}")
    result = await agent.ainvoke(
        {"messages": [HumanMessage(message)]},
        config={"recursion_limit": 2 * MAX_TOOL_ROUNDTRIPS + 4},
    )
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
