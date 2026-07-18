from .base import BaseAgent
from utils.web import search_internet
import datetime

class ResearcherAgent(BaseAgent):
    def run(self, user_input):
        # 1. Inject the current date so the model knows when to search for recent things
        current_year = datetime.datetime.now().year
        
        # 2. Give the model its ReAct instructions
        react_prompt = (
            f"Current Year: {current_year}\n\n"
            f"User request: {user_input}\n\n"
            "INSTRUCTIONS:\n"
            "- If you need to look up current documentation, papers, or facts, reply with ONLY the exact phrase: SEARCH: [your query here].\n"
            "- If you do not need to search, or if you have gathered enough information, just answer the question normally without the SEARCH prefix."
        )

        # Build the initial message array
        messages = self.build_messages(react_prompt)
        
        # 3. The Autonomous Loop (Max 3 searches to prevent infinite loops)
        max_loops = 3
        
        for iteration in range(max_loops):
            # Let the model think
            response = self.backend.generate(messages)
            
            # Did it decide to use the tool?
            if "SEARCH:" in response:
                # Extract the query (e.g., "SEARCH: PyTorch 2.0 release notes")
                query = response.split("SEARCH:")[1].strip()
                print(f"\n[Researcher] Autonomous Web Search: {query}")
                
                # Execute your DuckDuckGo utility
                web_results = search_internet(query)
                
                # Feed the observation back into the conversation history
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user", 
                    "content": f"OBSERVATION FROM WEB:\n{web_results}\n\nBased on this, either provide the final answer, or output another SEARCH: command."
                })
                
                # The loop restarts, allowing the model to generate the next step!
                continue
            
            # 4. If there is no SEARCH command, the model has decided it has the final answer
            else:
                self.history.append({"input": user_input, "output": response})
                self.update_memory(user_input, response)
                return response
                
        # 5. Failsafe: If it searches 3 times and still doesn't answer
        final_fallback = "I reached my maximum search limit. Here is what I found so far:\n\n" + response
        self.history.append({"input": user_input, "output": final_fallback})
        self.update_memory(user_input, final_fallback)
        return final_fallback