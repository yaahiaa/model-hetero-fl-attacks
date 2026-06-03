import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blockchain_compile import (
    SOLC_OPTIMIZE,
    SOLC_OPTIMIZE_RUNS,
    SOLC_VERSION,
    SOLC_VIA_IR,
    compile_commitment_ledger_contract,
)


def check_ipfs(api_url):
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError('Missing requests. Install with: pip install -r requirements-blockchain.txt') from exc

    api_url = api_url.rstrip('/')
    try:
        response = requests.post(f'{api_url}/api/v0/id', timeout=15)
        response.raise_for_status()
        node_id = response.json().get('ID', '<unknown>')
        print(f'IPFS API OK: {api_url} node_id={node_id}')
        content = b'commitment-backend-smoke-test\n'
        add_response = requests.post(
            f'{api_url}/api/v0/add',
            files={'file': ('smoke.txt', content, 'text/plain')},
            timeout=30,
        )
        add_response.raise_for_status()
        cid = add_response.json().get('Hash')
        if not cid:
            raise RuntimeError(f'IPFS add did not return Hash: {add_response.text}')
        cat_response = requests.post(f'{api_url}/api/v0/cat', params={'arg': cid}, timeout=30)
        cat_response.raise_for_status()
        if cat_response.content != content:
            raise RuntimeError(f'IPFS cat mismatch for cid={cid}')
        print(f'IPFS add/cat OK: cid={cid}')
    except requests.RequestException as exc:
        raise RuntimeError(f'IPFS API unavailable at {api_url}. Start Kubo with `ipfs daemon` after `ipfs init`.') from exc


def compile_contract():
    contract_path = SRC_DIR / 'contracts' / 'CommitmentLedger.sol'
    print(f'Contract path: {contract_path}')
    print(f'Solc version: {SOLC_VERSION}')
    print(f'optimize={SOLC_OPTIMIZE} optimize_runs={SOLC_OPTIMIZE_RUNS} via_ir={SOLC_VIA_IR}')
    abi, bytecode = compile_commitment_ledger_contract(contract_path)
    print('Contract compiled OK')
    return abi, bytecode


def check_blockchain(rpc_url, chain_id, account_index, deploy_contract):
    try:
        from web3 import Web3
    except ImportError as exc:
        raise RuntimeError('Missing web3. Install with: pip install -r requirements-blockchain.txt') from exc

    web3 = Web3(Web3.HTTPProvider(rpc_url))
    if not web3.is_connected():
        raise RuntimeError(f'Ethereum RPC unavailable at {rpc_url}. Start Ganache/Hardhat/Anvil on that URL.')
    print(f'Blockchain RPC OK: {rpc_url}')
    print(f'Configured chain id: {chain_id}')
    print(f'Node chain id: {web3.eth.chain_id}')
    accounts = list(web3.eth.accounts)
    print(f'Accounts ({len(accounts)}): {accounts}')
    print(f'Latest block: {web3.eth.block_number}')
    abi, bytecode = compile_contract()

    if deploy_contract:
        if account_index >= len(accounts):
            raise RuntimeError(f'Account index {account_index} unavailable; node returned {len(accounts)} accounts')
        account = accounts[account_index]
        contract = web3.eth.contract(abi=abi, bytecode=bytecode)
        started = time.perf_counter()
        tx_hash = contract.constructor().transact({'from': account, 'gas': 8000000, 'chainId': int(chain_id)})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if int(receipt.status) != 1:
            raise RuntimeError(f'Contract deployment transaction failed: tx={tx_hash.hex()}')
        print(
            'Contract deployed at '
            f'{receipt.contractAddress} tx={tx_hash.hex()} block={receipt.blockNumber} '
            f'gas_used={receipt.gasUsed} latency_sec={time.perf_counter() - started:.3f}'
        )


def main():
    parser = argparse.ArgumentParser(description='Check IPFS and Ethereum-compatible local backends.')
    parser.add_argument('--ipfs_api_url', default='http://127.0.0.1:5001')
    parser.add_argument('--blockchain_rpc_url', default='http://127.0.0.1:8545')
    parser.add_argument('--blockchain_chain_id', default=1337, type=int)
    parser.add_argument('--blockchain_account_index', default=0, type=int)
    parser.add_argument('--deploy_contract', action='store_true')
    args = parser.parse_args()
    check_ipfs(args.ipfs_api_url)
    check_blockchain(args.blockchain_rpc_url, args.blockchain_chain_id, args.blockchain_account_index, args.deploy_contract)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
