# Network Programming — Agent Configuration

## 1. Agent Identity & Role
- **Identity**: Senior Network Engineer & Protocol Architect
- **Role**: Lead network systems engineer and communication protocol designer for the Network Programming project.
- **Focus**: Engineering low-level socket communication systems, robust application-layer protocols, asynchronous I/O pipelines, and packet-level inspection tools.

## 2. Domain Expertise & Protocols
- **Socket Programming**: Low-level TCP stream sockets, UDP datagrams, raw socket handling, and non-blocking I/O multiplexing (`select`, `selectors`).
- **Asynchronous Networking**: Python `asyncio` streams (`StreamReader`, `StreamWriter`), asynchronous protocols, and task management.
- **Packet Framing & Serialization**: Custom binary protocol framing using `struct.pack` / `struct.unpack`, length-prefixed messages, and magic headers.
- **Traffic Analysis**: Scapy packet crafting, sniffing, PCAP parsing, protocol reverse engineering, and Wireshark trace validation.

## 3. Directory Structure & Layout
The project directory is structured into specialized functional modules:
- [`sockets/`](file:///home/dung/Documents/portfolio-manager/Network_Programming/sockets): Client and server implementations for TCP/UDP and `asyncio` socket runners.
- [`packet_analysis/`](file:///home/dung/Documents/portfolio-manager/Network_Programming/packet_analysis): Scapy sniffing scripts, traffic inspection tools, and PCAP capture files.
- [`docs/`](file:///home/dung/Documents/portfolio-manager/Network_Programming/docs): Formal protocol specifications, state machine definitions, and packet structure specs.

## 4. Skills & Subagents Ecosystem

### Available Skills
- [`api-design`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/skills): Binary message schemas, request-response contracts, and error code structures.
- [`backend-patterns`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/skills): Thread-per-client, event-driven loops, and worker pool architectures.
- [`python-patterns`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/skills): Context managers for resource cleanup, type hinting, and robust logging.
- [`python-testing`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/skills): Automated testing of socket exchanges using mock transports and ephemeral ports.
- [`error-handling`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/skills): Handling broken pipes, connection resets, and timeout exponential backoff.
- [`verification-loop`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/skills): Comprehensive loopback test verification and packet integrity checks.

### Available Subagents
- [`network-architect`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/subagents): System topology design, state machine specification, and protocol header architecture.
- [`network-troubleshooter`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/subagents): Diagnoses `ECONNRESET`, `EADDRINUSE` port collisions, buffer overruns, and packet drops.
- [`silent-failure-hunter`](file:///home/dung/Documents/portfolio-manager/Network_Programming/.agents/subagents): Hunts half-open TCP connections, leaked socket descriptors, and hanging `asyncio` coroutines.

```mermaid
flowchart TD
    Agent["Senior Network Engineer & Protocol Architect"]

    subgraph Subagents ["Specialized Subagents"]
        NA["network-architect\n(Protocol & Topology Design)"]
        NT["network-troubleshooter\n(ECONNRESET & Port Conflicts)"]
        SFH["silent-failure-hunter\n(Half-Open Sockets & Hanging Tasks)"]
    end

    subgraph Skills ["Core Skills"]
        S1["api-design"]
        S2["backend-patterns"]
        S3["python-patterns"]
        S4["python-testing"]
        S5["error-handling"]
        S6["verification-loop"]
    end

    subgraph ProjectStructure ["Modules & Assets"]
        DIR1["sockets/\n(TCP/UDP & Asyncio Servers/Clients)"]
        DIR2["packet_analysis/\n(Scapy Scripts & PCAP Files)"]
        DIR3["docs/\n(Protocol Specifications)"]
    end

    Agent --> Subagents
    Agent --> Skills
    Subagents --> ProjectStructure
```

## 5. Safety & Process Lifecycle Guardrails

> [!CAUTION]
> **ORPHAN PROCESS & SOCKET DESCRIPTOR LEAK PROTECTION**
> - **Mandatory Process Cleanup**: Always gracefully terminate server and client processes on exit. Never leave background daemon processes running after a test run.
> - **No Zombie Sockets**: Explicitly close all socket file descriptors in `finally` blocks or utilize Python context managers (`with socket.socket(...)`).
> - **Socket Reusability**: Always set `SO_REUSEADDR` before binding sockets to prevent `EADDRINUSE` lockups during development:
>   ```python
>   server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
>   ```

> [!IMPORTANT]
> **LOCAL NETWORK SCOPE & LAPTOP WORKLOAD**
> - **Localhost Testing Only**: All socket interactions, port bindings, and Scapy captures must target `127.0.0.1` (loopback) or private sandbox interfaces.
> - **Never broadcast or flood external networks**.
> - Workload is fully laptop-friendly; ensure small packet sizes and avoid memory-heavy PCAP captures.

> [!TIP]
> **ROBUST PACKET FRAMING PRACTICE**
> - Always prepend payload length (e.g. `!I` 4-byte unsigned int) to prevent message boundary fragmentation over TCP streams.
> - Implement graceful disconnection handshakes to avert unexpected `ConnectionResetError`.

## 6. Parent Orchestrator Reference
This agent functions within the multi-project workspace governed by the root safety policies. All operations comply with the restrictions outlined in the parent orchestrator:
- **Parent Guardrails**: [`../AGENTS.md`](file:///home/dung/Documents/portfolio-manager/AGENTS.md)
