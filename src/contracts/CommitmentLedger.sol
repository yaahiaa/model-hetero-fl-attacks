// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CommitmentLedger {
    struct Candidate {
        bool exists;
        uint256 roundId;
        bytes32 previousParentHash;
        string previousParentCid;
        bytes32 candidateParentHash;
        string candidateCid;
        bytes32 artifactSha256;
        uint256 artifactSizeBytes;
        string scheduleId;
        uint256[] activeCohortsScaled;
        uint256[] committeeIds;
        uint256 timestamp;
        address submitter;
        bytes32 metadataHash;
    }

    struct Report {
        bool exists;
        uint256 roundId;
        bytes32 candidateParentHash;
        uint256 verifierUserId;
        uint256 cohortRateScaled;
        bool approved;
        uint8 reasonCode;
        uint16 failedChecksBitmask;
        uint256 relativeChangeScaled;
        bytes32 attestationHash;
        address submitter;
        uint256 timestamp;
    }

    struct Decision {
        bool exists;
        uint256 roundId;
        bool approved;
        string reason;
        uint256 numApproved;
        uint256 numRejected;
        string quorumRule;
        bytes32 approvedParentHash;
        string approvedParentCid;
        uint256 timestamp;
    }

    bytes32 public latestApprovedParentHash;
    string public latestApprovedParentCid;
    bytes32 public latestApprovedArtifactSha256;
    uint256 public latestApprovedArtifactSizeBytes;
    bool public hasLatestApprovedParent;

    mapping(uint256 => Candidate) private candidates;
    mapping(uint256 => Report[]) private reportsByRound;
    mapping(uint256 => Decision) private decisions;

    event GenesisCommitted(
        bytes32 parentHash,
        string cid,
        bytes32 artifactSha256,
        uint256 artifactSizeBytes,
        bytes32 metadataHash,
        address submitter,
        uint256 timestamp
    );
    event CandidateSubmitted(uint256 roundId, bytes32 previousParentHash, bytes32 candidateParentHash, string cid, address submitter);
    event VerificationReportSubmitted(
        uint256 indexed roundId,
        bytes32 indexed candidateParentHash,
        uint256 indexed verifierUserId,
        uint256 cohortRateScaled,
        bool approved,
        uint8 reasonCode,
        uint16 failedChecksBitmask,
        uint256 relativeChangeScaled,
        address submitter
    );
    event RoundFinalized(uint256 roundId, bool approved, uint256 numApproved, uint256 numRejected, string quorumRule);

    function commitGenesisParent(
        bytes32 parentHash,
        string calldata cid,
        bytes32 artifactSha256,
        uint256 artifactSizeBytes,
        bytes32 metadataHash
    ) external {
        require(!hasLatestApprovedParent, "genesis already committed");
        latestApprovedParentHash = parentHash;
        latestApprovedParentCid = cid;
        latestApprovedArtifactSha256 = artifactSha256;
        latestApprovedArtifactSizeBytes = artifactSizeBytes;
        hasLatestApprovedParent = true;
        emit GenesisCommitted(parentHash, cid, artifactSha256, artifactSizeBytes, metadataHash, msg.sender, block.timestamp);
    }

    function submitCandidate(
        uint256 roundId,
        bytes32 previousParentHash,
        string calldata previousParentCid,
        bytes32 candidateParentHash,
        string calldata cid,
        bytes32 artifactSha256,
        uint256 artifactSizeBytes,
        string calldata scheduleId,
        uint256[] calldata activeCohortsScaled,
        uint256[] calldata committeeIds,
        bytes32 metadataHash
    ) external {
        require(hasLatestApprovedParent, "missing genesis parent");
        require(previousParentHash == latestApprovedParentHash, "previous parent mismatch");

        Candidate storage candidate = candidates[roundId];
        delete candidate.activeCohortsScaled;
        delete candidate.committeeIds;

        candidate.exists = true;
        candidate.roundId = roundId;
        candidate.previousParentHash = previousParentHash;
        candidate.previousParentCid = previousParentCid;
        candidate.candidateParentHash = candidateParentHash;
        candidate.candidateCid = cid;
        candidate.artifactSha256 = artifactSha256;
        candidate.artifactSizeBytes = artifactSizeBytes;
        candidate.scheduleId = scheduleId;
        candidate.timestamp = block.timestamp;
        candidate.submitter = msg.sender;
        candidate.metadataHash = metadataHash;

        for (uint256 i = 0; i < activeCohortsScaled.length; i++) {
            candidate.activeCohortsScaled.push(activeCohortsScaled[i]);
        }
        for (uint256 i = 0; i < committeeIds.length; i++) {
            candidate.committeeIds.push(committeeIds[i]);
        }

        delete reportsByRound[roundId];
        delete decisions[roundId];
        emit CandidateSubmitted(roundId, previousParentHash, candidateParentHash, cid, msg.sender);
    }

    function submitVerificationReport(
        uint256 roundId,
        bytes32 candidateParentHash,
        uint256 verifierUserId,
        uint256 cohortRateScaled,
        bool approved,
        uint8 reasonCode,
        uint16 failedChecksBitmask,
        uint256 relativeChangeScaled
    ) external {
        require(candidates[roundId].exists, "missing candidate");
        require(candidates[roundId].candidateParentHash == candidateParentHash, "candidate hash mismatch");
        bytes32 attestationHash = keccak256(abi.encode(
            roundId,
            candidateParentHash,
            verifierUserId,
            cohortRateScaled,
            approved,
            reasonCode,
            failedChecksBitmask,
            relativeChangeScaled
        ));
        reportsByRound[roundId].push(Report({
            exists: true,
            roundId: roundId,
            candidateParentHash: candidateParentHash,
            verifierUserId: verifierUserId,
            cohortRateScaled: cohortRateScaled,
            approved: approved,
            reasonCode: reasonCode,
            failedChecksBitmask: failedChecksBitmask,
            relativeChangeScaled: relativeChangeScaled,
            attestationHash: attestationHash,
            submitter: msg.sender,
            timestamp: block.timestamp
        }));
        emit VerificationReportSubmitted(
            roundId,
            candidateParentHash,
            verifierUserId,
            cohortRateScaled,
            approved,
            reasonCode,
            failedChecksBitmask,
            relativeChangeScaled,
            msg.sender
        );
    }

    function finalizeRound(uint256 roundId, string calldata quorumRule) external {
        Candidate storage candidate = candidates[roundId];
        require(candidate.exists, "missing candidate");

        Report[] storage roundReports = reportsByRound[roundId];
        uint256 numApproved = 0;
        uint256 numRejected = 0;
        for (uint256 i = 0; i < roundReports.length; i++) {
            if (roundReports[i].approved) {
                numApproved += 1;
            } else {
                numRejected += 1;
            }
        }

        bytes32 ruleHash = keccak256(bytes(quorumRule));
        bool approved;
        string memory reason;
        if (ruleHash == keccak256(bytes("majority"))) {
            approved = roundReports.length > 0 && numApproved > numRejected;
            reason = approved ? "approved_by_majority" : "rejected_by_majority";
        } else if (ruleHash == keccak256(bytes("committee_unanimous"))) {
            approved = roundReports.length > 0 && numRejected == 0;
            reason = approved ? "approved_all_committee" : "rejected_committee_dissent";
        } else {
            revert("unsupported quorum rule");
        }

        bytes32 approvedParentHash = bytes32(0);
        string memory approvedParentCid = "";
        if (approved) {
            approvedParentHash = candidate.candidateParentHash;
            approvedParentCid = candidate.candidateCid;
            latestApprovedParentHash = candidate.candidateParentHash;
            latestApprovedParentCid = candidate.candidateCid;
            latestApprovedArtifactSha256 = candidate.artifactSha256;
            latestApprovedArtifactSizeBytes = candidate.artifactSizeBytes;
            hasLatestApprovedParent = true;
        }

        decisions[roundId] = Decision({
            exists: true,
            roundId: roundId,
            approved: approved,
            reason: reason,
            numApproved: numApproved,
            numRejected: numRejected,
            quorumRule: quorumRule,
            approvedParentHash: approvedParentHash,
            approvedParentCid: approvedParentCid,
            timestamp: block.timestamp
        });

        emit RoundFinalized(roundId, approved, numApproved, numRejected, quorumRule);
    }

    function getCandidate(uint256 roundId) external view returns (Candidate memory) {
        return candidates[roundId];
    }

    function getReports(uint256 roundId) external view returns (Report[] memory) {
        return reportsByRound[roundId];
    }

    function getDecision(uint256 roundId) external view returns (Decision memory) {
        return decisions[roundId];
    }

    function getLatestApprovedParent() external view returns (
        bool exists,
        bytes32 parentHash,
        string memory cid,
        bytes32 artifactSha256,
        uint256 artifactSizeBytes
    ) {
        return (
            hasLatestApprovedParent,
            latestApprovedParentHash,
            latestApprovedParentCid,
            latestApprovedArtifactSha256,
            latestApprovedArtifactSizeBytes
        );
    }
}
