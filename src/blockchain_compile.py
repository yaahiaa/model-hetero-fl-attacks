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

    installed_versions = {str(version) for version in solcx.get_installed_solc_versions()}
    if SOLC_VERSION not in installed_versions:
        raise RuntimeError(
            f'Solidity compiler {SOLC_VERSION} is not installed. Install it with: '
            f'python -c "import solcx; solcx.install_solc(\'{SOLC_VERSION}\')"'
        )

    try:
        solcx.set_solc_version(SOLC_VERSION)
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
        raise RuntimeError(f'Compilation succeeded, but CommitmentLedger was not found in {contract_path}')
    return compiled[contract_key]['abi'], compiled[contract_key]['bin']
