# Scientific support agentic workflow
![Agentic workflow](./docs/Agentic%20workflow%20visualized.png)
Note that this is just a ChatGPT generated placeholder, later needs to be modified.

## TODOS:
- Designer agent for perfect coding.
- planner->agent->critiq->agent->response
    - General Agent loop for all agents. I don't want to see as they explain everything about the question I ask.
- Skills and other md.
- Add other agents if needed.
- Add time logging, for fund summary.

## TO RUN THE PROJECT

- Clone the repository and install dependencies
- sqlite3 chainlit_chats.db < schema.sql
- Create a chainlit API
    ```
    chainlit create-secret
    ```
- Copy the generated random string and paste into the `.env` file
