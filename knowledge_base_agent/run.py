import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.schemas import AgentRunInput, AgentDeployment, KBRunInput
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from knowledge_base_agent.schemas import (
    InputSchema, 
    QueryInput, 
    StoreInput
)

load_dotenv()

logger = get_logger(__name__)

class KnowledgeBaseAgent:
    """ A knowledge base agent that interfaces with Market_kb through naptha-sdk """

    def __init__(self, deployment: AgentDeployment, consumer_id: str):
        self.deployment = deployment
        self.consumer_id = consumer_id
        self.market_kb = KnowledgeBase()
    
    def _create_kb_input(self, func_name: str, signature: str, func_input_data: Optional[Dict[str, Any]] = None) -> KBRunInput:
        """ Helper method to create KBRunInput with proper signature """

        return KBRunInput(
            consumer_id=self.consumer_id,
            inputs={
                "func_name": func_name,
                "func_input_data": func_input_data
            },
            deployment=self.deployment.kb_deployments[0],
            signature=signature
        )
    
    async def store(self, model_run: Dict[str, Any]) -> Dict[str, Any]:
        """ Store knowledge in the market knowledge base """

        try:
            store_input = StoreInput(**model_run.inputs.func_input_data)
            kb_run_input = self._create_kb_input(
                func_name="ingest_knowledge",
                func_input_data=store_input.model_dump(),
                signature=model_run.signature
            )
            result = await self.market_kb.run(kb_run_input)

            return result.model_dump()
        except Exception as e:
            logger.error(f"Error storing knowledge: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def query(self, model_run: Dict[str, Any]) -> Dict[str, Any]:
        """ Query the knowledge base """

        try:
            query_input = QueryInput(**model_run.inputs.func_input_data)
            logger.info(f"Querying KB with: {query_input.query}")
            
            kb_query_data = {
                "query": query_input.query,
                "top_k": query_input.top_k
            }
            
            # Set up and execute the KB query
            kb_run_input = self._create_kb_input(
                func_name="search",
                func_input_data=kb_query_data,
                signature=model_run.signature
            )

            await self.market_kb.create(self.deployment.kb_deployments[0])
            kb_response = await self.market_kb.run(kb_run_input)
            
            if hasattr(kb_response, "model_dump"):
                response_data = kb_response.model_dump()
            else:
                response_data = kb_response

            if response_data.get("status") == "completed" and "results" in response_data:
                results = json.loads(response_data["results"][0])["results"]
                data = json.loads(results[0])['data']
                if not data:
                    return {
                        "status": "success", 
                        "message": "No relevant information found for your query.",
                        "results": []
                    }
                    
                return {
                    "status": "success",
                    "results": json.dumps(data)
                }
           
            else:
                return {
                    "status": response_data.get("status", "error"),
                    "message": response_data.get("message", "Unknown error in knowledge base query"),
                    "results": []
                }
                
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return {
                "status": "error", 
                "message": f"Error querying knowledge base: {str(e)}",
                "results": []
            }

    async def clear(self, model_run: Dict[str, Any]) -> Dict[str, Any]:
        """ Clear all data from the knowledge base """

        try:
            kb_run_input = self._create_kb_input(func_name="clear", signature=model_run.signature)
            result = await self.market_kb.run(kb_run_input)
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return {"status": "error", "message": str(e)}

async def run(module_run: Dict[str, Any]) -> Dict[str, Any]:
    """ Run the Knowledge Base Agent deployment """

    try:
        module_run = AgentRunInput(**module_run)
        module_run.inputs = InputSchema(**module_run.inputs)
        agent = KnowledgeBaseAgent(module_run.deployment, module_run.consumer_id)
        await agent.market_kb.create(deployment=module_run.deployment.kb_deployments[0])
        
        method = getattr(agent, module_run.inputs.func_name)
        if not method:
            raise ValueError(f"Invalid function name: {module_run.inputs.func_name}")
            
        result = await method(module_run)
        return result
    except Exception as e:
        logger.error(f"Error running knowledge base agent: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import asyncio
    import json
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()

    # Testing code courtesy by OpenAI
    async def test_agent():
        print("\n" + "="*80)
        print("🧠 Testing Knowledge Base Agent".center(80))
        print("="*80 + "\n")

        try:
            deployment = await setup_module_deployment(
                "agent",
                "knowledge_base_agent/configs/deployment.json",
                node_url=os.getenv("NODE_URL")
            )
        except Exception as e:
            print("❌ Deployment Error:", str(e))
            return

        # Example store operation
        print("📝 Testing Knowledge Storage".center(80))
        print("-"*80)
        
        # Clean, single-line text without extra whitespace
        test_text = "Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage."
        
        store_input = {
            "inputs": {
                "func_name": "store",
                "func_input_data": {
                    "content": test_text,
                    "metadata": {"source": "Lorem Ipsum History", "type": "test"}
                }
            },
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        }

        result = await run(store_input)
        if result.get("status") == "error":
            error_msg = result.get("error_message") or result.get("message") or "Unknown error occurred"
            print("❌ Storage Error:", error_msg)
        else:
            knowledge_id = result.get("data", {}).get("id") or result.get("id")
            print("✅ Successfully stored knowledge with ID:", knowledge_id or "N/A")
        
        print("\n" + "-"*80)
        
        # Example query operation
        print("🔍 Testing Knowledge Query".center(80))
        print("-"*80)
        
        query_input = {
            "inputs": {
                "func_name": "query",
                "func_input_data": {
                    "query": "What is Lorem Ipsum?",
                    "top_k": 2
                }
            },
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        }

        result = await run(query_input)
        if result.get("error"):
            error_msg = result.get("error_message") or "Unknown error occurred"
            print("❌ Query Error:", error_msg)
        else:
            try:
                # Parse the JSON string from results array
                if result.get("status") == "success":
                    data = json.loads(result.get('results')) if isinstance(result.get('results'), str) else result.get('results')
                    print("✅ Found", len(data), "relevant results\n")
                    
                    for i, item in enumerate(data, 1):
                        print(f"\n📎 Result {i}:")
                        print("="*40)
                        
                        # Content
                        chunk = item.get("chunk", "").strip()
                        print("📄 Content:".ljust(12), chunk[:200] + "..." if len(chunk) > 200 else chunk)
                        
                        # Metadata
                        metadata = item.get("metadata", {})
                        if metadata:
                            print("\n🏷️  Metadata:")
                            for key, value in metadata.items():
                                print(" "*12 + f"{key}: {value}")
                        
                        # Source and Timestamp
                        print("\n📚 Source:".ljust(12), item.get("source") or "Unknown")
                        print("⌚ Time:".ljust(12), item.get("timestamp", "N/A"))
                        
                        if i < len(data):
                            print("\n" + "-"*80)
                else:
                    print("❌ Knowledge Base Error:", result.get("message", "Unknown error"))
            except json.JSONDecodeError as e:
                print("❌ Error parsing results:", str(e))
            except Exception as e:
                print("❌ Error processing results:", str(e))

        print("\n" + "="*80)
        print("✨ Test Complete".center(80))
        print("="*80)

    asyncio.run(test_agent())