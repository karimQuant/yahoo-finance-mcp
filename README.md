# yahoo-finance-mcp

[![smithery badge](https://smithery.ai/badge/@Alex2Yang97/yahoo-finance-mcp)](https://smithery.ai/server/@Alex2Yang97/yahoo-finance-mcp)

<h2 align="center">⚠️ This project is currently in progress. Installation may not fully function.</h2>

Introducing the Yahoo Finance MCP Server: a robust tool designed to provide seamless integration between LLMs like Claude and Yahoo Finance for retrieving real-time and historical financial data. The server can be utilized to fetch a wide range of financial information, including, but not limited to, stock prices, historical financial data, exchange rates, dividend history, and more. 
 
## Getting Started

### Installing via Smithery

To install yahoo-finance-mcp for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@Alex2Yang97/yahoo-finance-mcp):

```bash
npx -y @smithery/cli install @Alex2Yang97/yahoo-finance-mcp --client claude
```

1. **Clone the repository**

   Start by cloning the repository to your local machine using:

   ```
   git clone https://github.com/Alex2Yang97/yahoo-finance-mcp.git
   ```

3. **Install dependencies**

   Navigate to the project's root directory and install the required dependencies. You can do this using yarn or npm. For example, if you're using npm, you'd run:

   ```
   npm install
   ```

4. **Configuration**

   Configure your MCP server with any necessary environment variables if applicable. Refer to the documentation for any specific setup instructions.

### How to Use

To use the Yahoo Finance MCP Server, start the server using the following command:

```
node index.js
```

Once the server is up and running, you can connect your LLM or other HTTP clients to it using the appropriate localhost address and port number (for example, http://localhost:4000). Here, you'll have access to a suite of functionalities to obtain financial data from Yahoo Finance by using the defined JSON-RPC calls.

### Maintainer 
This versatile tool is built by...

...
