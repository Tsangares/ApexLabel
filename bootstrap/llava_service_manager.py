#!/usr/bin/env python3
"""
LLaVA Service Manager for Adaptive Bootstrap

This script ensures that Ollama with LLaVA models are running on the required ports
for dual GPU processing in step4_adaptive_bootstrap.py

Usage:
    python step0_start_llava.py [--check-only]
    
Ports:
    11434 - GPU 0 (CUDA_VISIBLE_DEVICES=0)
    11435 - GPU 1 (CUDA_VISIBLE_DEVICES=1)
"""

import subprocess
import requests
import time
import logging
import argparse
import os
import signal
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLaVAServiceManager:
    def __init__(self):
        self.ports = [11434, 11435]
        self.gpu_devices = [0, 1]
        self.model = "llava:7b"
        self.processes = {}
        
    def check_port_status(self, port: int) -> bool:
        """Check if Ollama is running and responding on given port"""
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=3)
            if response.status_code == 200:
                # Check if llava model is available
                models = response.json().get('models', [])
                llava_available = any('llava' in model.get('name', '') for model in models)
                if llava_available:
                    logger.info(f"✅ Port {port}: Ollama running with LLaVA model")
                    return True
                else:
                    logger.warning(f"⚠️  Port {port}: Ollama running but LLaVA model not found")
                    return False
            else:
                logger.warning(f"❌ Port {port}: Ollama responded with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.info(f"❌ Port {port}: No response - {str(e)}")
            return False
    
    def find_existing_processes(self):
        """Find existing Ollama processes on our target ports"""
        existing = {}
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'environ']):
            try:
                if proc.info['name'] == 'ollama' and 'serve' in proc.info['cmdline']:
                    # Check if this process has OLLAMA_HOST set to one of our ports
                    try:
                        env = proc.info.get('environ', {})
                        if env and 'OLLAMA_HOST' in env:
                            ollama_host = env['OLLAMA_HOST']
                            if ollama_host:  # Check if ollama_host is not None
                                for port in self.ports:
                                    if f':{port}' in ollama_host:
                                        existing[port] = proc.info['pid']
                                        logger.info(f"Found existing Ollama process on port {port} (PID: {proc.info['pid']})")
                        elif not env or 'OLLAMA_HOST' not in env:
                            # Default port 11434 if no OLLAMA_HOST is set
                            if 11434 in self.ports:
                                existing[11434] = proc.info['pid']
                                logger.info(f"Found existing Ollama process on default port 11434 (PID: {proc.info['pid']})")
                    except (psutil.AccessDenied, KeyError):
                        # If we can't read environment, assume it might be using default port
                        if 11434 in self.ports:
                            existing[11434] = proc.info['pid']
                            logger.info(f"Found existing Ollama process (cannot read env, assuming port 11434) (PID: {proc.info['pid']})")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return existing
    
    def kill_process_on_port(self, port: int):
        """Kill existing Ollama process on specific port"""
        existing = self.find_existing_processes()
        if port in existing:
            try:
                pid = existing[port]
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                logger.info(f"Terminated existing process on port {port} (PID: {pid})")
            except ProcessLookupError:
                logger.info(f"Process on port {port} already terminated")
            except Exception as e:
                logger.error(f"Error terminating process on port {port}: {e}")
    
    def start_ollama_service(self, port: int, gpu_device: int) -> subprocess.Popen:
        """Start Ollama service on specific port and GPU"""
        logger.info(f"Starting Ollama on port {port} with GPU {gpu_device}...")
        
        # Kill any existing process on this port first
        self.kill_process_on_port(port)
        
        # Set environment for specific GPU and port
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'  # Ollama uses OLLAMA_HOST env var, not --port flag
        
        # Start Ollama serve
        cmd = ['ollama', 'serve']
        
        try:
            # Start process detached so it survives parent exit
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent session
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✅ Ollama started on port {port} (PID: {process.pid})")
                return process
            else:
                logger.error(f"❌ Ollama failed to start on port {port}")
                logger.error(f"Process exited with code: {process.returncode}")
                return None
                
        except FileNotFoundError:
            logger.error("❌ Ollama not found in PATH. Please install Ollama first.")
            return None
        except Exception as e:
            logger.error(f"❌ Error starting Ollama on port {port}: {e}")
            return None
    
    def ensure_llava_model(self, port: int):
        """Ensure LLaVA model is pulled and available"""
        logger.info(f"Checking LLaVA model availability on port {port}...")
        
        # Wait for service to be ready
        max_retries = 30
        for i in range(max_retries):
            if self.check_port_status(port):
                return True
            
            if i == 0:
                logger.info(f"Pulling LLaVA model on port {port}...")
                try:
                    # Pull the model
                    result = subprocess.run(
                        ['ollama', 'pull', self.model],
                        env={**os.environ, 'OLLAMA_HOST': f'localhost:{port}'},
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout for model pull
                    )
                    if result.returncode == 0:
                        logger.info(f"✅ LLaVA model pulled successfully on port {port}")
                    else:
                        logger.error(f"❌ Failed to pull LLaVA model on port {port}: {result.stderr}")
                        return False
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ Timeout pulling LLaVA model on port {port}")
                    return False
            
            time.sleep(2)
        
        logger.error(f"❌ LLaVA model not available on port {port} after {max_retries} retries")
        return False
    
    def check_all_services(self) -> dict:
        """Check status of all required services"""
        status = {}
        for port in self.ports:
            status[port] = self.check_port_status(port)
        return status
    
    def start_all_services(self) -> bool:
        """Start all required LLaVA services"""
        logger.info("🚀 Starting LLaVA services for dual GPU processing...")
        
        success = True
        for port, gpu_device in zip(self.ports, self.gpu_devices):
            # Start Ollama service
            process = self.start_ollama_service(port, gpu_device)
            if process:
                self.processes[port] = process
                
                # Ensure LLaVA model is available
                if not self.ensure_llava_model(port):
                    success = False
            else:
                success = False
        
        if success:
            logger.info("✅ All LLaVA services started successfully!")
            self.print_status()
        else:
            logger.error("❌ Failed to start all LLaVA services")
            
        return success
    
    def print_status(self):
        """Print current status of all services"""
        logger.info("📊 LLaVA Service Status:")
        status = self.check_all_services()
        for port in self.ports:
            gpu_device = self.gpu_devices[self.ports.index(port)]
            status_emoji = "✅" if status[port] else "❌"
            logger.info(f"  {status_emoji} Port {port} (GPU {gpu_device}): {'Ready' if status[port] else 'Not Ready'}")
    
    def stop_all_services(self):
        """Stop all managed services"""
        logger.info("🛑 Stopping LLaVA services...")
        for port, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ Stopped service on port {port}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.info(f"🔪 Force-killed service on port {port}")
            except Exception as e:
                logger.error(f"❌ Error stopping service on port {port}: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'processes'):
            self.stop_all_services()

def main():
    parser = argparse.ArgumentParser(description="LLaVA Service Manager")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check status, don't start services")
    parser.add_argument("--stop", action="store_true",
                       help="Stop all LLaVA services")
    
    args = parser.parse_args()
    
    manager = LLaVAServiceManager()
    
    if args.stop:
        existing = manager.find_existing_processes()
        for port in manager.ports:
            if port in existing:
                manager.kill_process_on_port(port)
        logger.info("✅ Stopped all LLaVA services")
        return True
    
    if args.check_only:
        manager.print_status()
        status = manager.check_all_services()
        return all(status.values())
    
    # Check current status first
    status = manager.check_all_services()
    all_ready = all(status.values())
    
    if all_ready:
        logger.info("✅ All LLaVA services already running and ready!")
        manager.print_status()
        return True
    else:
        logger.info("⚠️  Some LLaVA services need to be started...")
        return manager.start_all_services()

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        exit(1)