from pathlib import Path
from typing import Any, Dict, Tuple


SOLC_VERSION = '0.8.24'
SOLC_OPTIMIZE = True
SOLC_OPTIMIZE_RUNS = 200
SOLC_VIA_IR = True
SOLC_OUTPUT_VALUES = ['abi', 'bin']


def compile_commitment_ledger_contract(contract_path: Path) -> Tuple[Any, str]:
    try:
        import solcx
    except ImportError as exc:
        raise RuntimeError(
            'Missing py-solc-x. Install optional blockchain dependencies with: '
            'pip install -r requirements-blockchain.txt'
        ) from exc

    contract_path = Path(contract_path).resolve()
    if not contract_path.exists():
        raise RuntimeError(f'Solidity contract not found: {contract_path}')

    try:
        installed_versions = {str(version) for version in solcx.get_installed_solc_versions()}

        if SOLC_VERSION not in installed_versions:
            print(
                f'Solidity compiler {SOLC_VERSION} not found in py-solc-x. '
                f'Installed versions before install: {sorted(installed_versions)}'
            )
            print(f'Installing solc {SOLC_VERSION}...')
            solcx.install_solc(SOLC_VERSION)

        installed_versions = {str(version) for version in solcx.get_installed_solc_versions()}
        if SOLC_VERSION not in installed_versions:
            raise RuntimeError(
                f'Solidity compiler {SOLC_VERSION} is still not visible after install. '
                f'Installed versions: {sorted(installed_versions)}'
            )

        solcx.set_solc_version(SOLC_VERSION)

        print(f'Contract path: {contract_path}')
        print(f'Solc version: {SOLC_VERSION}')
        print(
            f'optimize={SOLC_OPTIMIZE} '
            f'optimize_runs={SOLC_OPTIMIZE_RUNS} '
            f'via_ir={SOLC_VIA_IR}'
        )

        source = contract_path.read_text(encoding='utf-8')

        compiled: Dict[str, Dict[str, Any]] = solcx.compile_source(
            source,
            output_values=SOLC_OUTPUT_VALUES,
            solc_version=SOLC_VERSION,
            optimize=SOLC_OPTIMIZE,
            optimize_runs=SOLC_OPTIMIZE_RUNS,
            via_ir=SOLC_VIA_IR,
        )

    except Exception as exc:
        raise RuntimeError(
            f'Could not compile {contract_path} with solc={SOLC_VERSION}, '
            f'optimize={SOLC_OPTIMIZE}, optimize_runs={SOLC_OPTIMIZE_RUNS}, '
            f'via_ir={SOLC_VIA_IR}. Original error: {exc}'
        ) from exc

    contract_key = next((key for key in compiled if key.endswith(':CommitmentLedger')), None)

    if contract_key is None:
        raise RuntimeError(
            f'Compilation succeeded, but CommitmentLedger was not found in {contract_path}. '
            f'Compiled contracts: {list(compiled.keys())}'
        )

    print('Contract compiled OK')

    return compiled[contract_key]['abi'], compiled[contract_key]['bin']
