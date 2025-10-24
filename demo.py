"""
Interactive demo of LLM + SLM routing system
Captures and displays the complete thought process
"""
import asyncio
import sys
import re
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Global to capture output
captured_logs = []

class LogCapture:
    """Capture print statements from router_agent"""
    def __init__(self):
        self.logs = []
        self.current_call = None
    
    def write(self, text):
        if text.strip():
            # Filter out TensorFlow/CUDA warnings
            if any(skip in text for skip in [
                'external/local_xla',
                'Unable to register',
                'cuFFT', 'cuDNN', 'cuBLAS',
                'computation placer',
                'TensorFlow binary is optimized',
                'absl::InitializeLog'
            ]):
                return  # Skip these warnings
            
            self.logs.append(text)
            
            # Parse for tool calls - FIX: Match actual print from tool
            if '[TOOL] slm_help:' in text:
                question = text.split('slm_help:', 1)[1].strip() if 'slm_help:' in text else text
                self.current_call = {'question': question, 'output': '', 'answer': ''}
            
            elif '[SLM OUTPUT] Full response:' in text:
                if self.current_call:
                    # Capture everything after the marker
                    remaining = text.split('Full response:', 1)[1].strip() if 'Full response:' in text else ''
                    self.current_call['output'] = remaining
            
            elif '[SLM RESULT] Extracted answer:' in text:
                if self.current_call:
                    self.current_call['answer'] = text.split('answer:', 1)[1].strip() if 'answer:' in text else ''
                    captured_logs.append(self.current_call.copy())
                    self.current_call = None
    
    def flush(self):
        pass

async def demo_problem(problem: str):
    """Run a problem and show the complete workflow"""
    global captured_logs
    captured_logs = []
    
    console.print("\n" + "="*80, style="bold blue")
    console.print(Panel(
        Text(problem, style="bold cyan"),
        title="[bold]Problem[/bold]",
        border_style="cyan"
    ))
    
    console.print("\n[bold yellow]🤖 LLM is analyzing the problem...[/bold yellow]\n")
    
    # Import here to capture prints
    from router_agent_demo import run_agent
    
    # Capture output while running
    log_capture = LogCapture()
    
    # Redirect prints temporarily
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log_capture
    sys.stderr = log_capture  # Also capture stderr
    
    try:
        result = await run_agent(problem)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Debug: Show all captured logs
        console.print(f"[dim]Debug: Captured {len(log_capture.logs)} log lines[/dim]")
        
        # Display captured tool calls
        if captured_logs:
            console.print(f"[bold green]🔧 Tool Calls: {len(captured_logs)}[/bold green]\n")
            
            for i, call in enumerate(captured_logs, 1):
                console.print(f"[bold magenta]═══ Tool Call #{i} ═══[/bold magenta]\n")
                
                # LLM → SLM
                console.print(Panel(
                    Text(call['question'], style="cyan"),
                    title="[cyan]LLM asks SLM[/cyan]",
                    border_style="blue"
                ))
                
                # SLM reasoning (truncated)
                reasoning = call['output'][:400] + '...' if len(call['output']) > 400 else call['output']
                console.print(Panel(
                    Text(reasoning, style="yellow"),
                    title="[yellow]SLM thinks[/yellow]",
                    border_style="yellow"
                ))
                
                # SLM → LLM
                console.print(Panel(
                    Text(f"Answer: {call['answer']}", style="green bold"),
                    title="[green]SLM returns to LLM[/green]",
                    border_style="green"
                ))
                console.print()
        else:
            console.print("[yellow]⚠️  No tool calls detected - LLM may have solved directly or tool wasn't invoked[/yellow]")
            console.print("[dim]Debug: Check if the agent's instructions are being followed[/dim]\n")
        
        # Final answer
        console.print(Panel(
            Text(result, style="bold green"),
            title="[bold green]Final Answer[/bold green]",
            border_style="bold green"
        ))
        
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}", style="bold red")
        import traceback
        traceback.print_exc()

async def interactive_demo():
    """Interactive demo mode"""
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]🚀 LLM + SLM Routing Demo[/bold cyan]\n\n"
            "See how GPT-4o delegates math tasks to Qwen-Math-1.5B"
        ),
        border_style="cyan"
    ))
    
    examples = [
        "What is 156 + 243?",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value by 150%. How much profit did he make?",
    ]
    
    while True:
        console.print("\n[bold]Choose:[/bold]")
        console.print("1. One-step Computation")
        console.print("2. Multi-step Problem")
        console.print("3. Complex Multi-step Problem")
        console.print("4. Your own problem")
        console.print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice in ['1', '2', '3']:
            await demo_problem(examples[int(choice) - 1])
        elif choice == '4':
            custom = input("\nEnter problem: ").strip()
            if custom:
                await demo_problem(custom)  
        elif choice == '5':
            console.print("\n[bold green]Thanks! 👋[/bold green]")
            break         
        else: 
            console.print("[red]Invalid choice[/red]")   

if __name__ == "__main__":
    try:
        from rich.console import Console
    except ImportError:
        print("Install rich: pip install rich")
        sys.exit(1)
    
    asyncio.run(interactive_demo())