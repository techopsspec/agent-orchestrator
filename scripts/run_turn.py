"""Manual driver for one agent turn -- for iterating on the agent wiring directly, without
the FastAPI layer. Requires an identity token minted by Rails, e.g.:

    cd Operator-Portal/backend && RAILS_ENV=development bin/rails runner \\
      'puts McpTools::IdentityToken.mint(user: User.find(1), conversation: ChatConversation.first)'

or, for the recruiting SMS bot:

    bin/rails runner 'puts McpTools::SmsIdentityToken.mint(conversation: SmsConversation.last)'

Usage:
    IDENTITY_TOKEN=... python scripts/run_turn.py "What snowfall zones exist?"
    IDENTITY_TOKEN=... python scripts/run_turn.py --app recruiting_sms --locale es "¿Cuánto pagan?"
"""

import argparse
import asyncio
import os

from langchain_core.messages import HumanMessage

from agent_orchestrator.apps.registry import APPS


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?", default="What snowfall zones exist?")
    parser.add_argument("--app", default="operator_portal", choices=sorted(APPS))
    parser.add_argument("--locale", default=None)
    args = parser.parse_args()

    app = APPS[args.app]
    agent, provider_map = await app.build_agent(os.environ["IDENTITY_TOKEN"], locale=args.locale)
    print(f"provider map: {provider_map}")
    result = await agent.ainvoke(
        {"messages": [HumanMessage(args.message)]},
        config={"recursion_limit": 2 * app.MAX_TOOL_ROUNDTRIPS + 4},
    )
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
