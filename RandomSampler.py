import random
import matplotlib.pyplot as plt
from typing import List, Tuple

def computePmf(p: List[int]) -> List[float]:
    """Compute the probability mass function (PMF) from a list of counts."""
    total = sum(p)
    return [value / total for value in p]

def computeCmf(pmf: List[float]) -> List[float]:
    """Compute the cumulative mass function (CMF) from a PMF."""

    cmf = []
    cmfSum = 0
    for pmf_value in pmf:
        cmfSum += pmf_value
        cmf.append(cmfSum)
    return cmf

def findClosestCmf(cmf: List[float], randomValue: float) -> Tuple[float, int]:
    """Find the closest CMF value greater than the random value and its index."""
    bucketIdx = 0
    for index, cmf_value in enumerate(cmf):
        if cmf_value > randomValue:
            bucketIdx = index
            return cmf_value, bucketIdx
    return None, bucketIdx  # Return None if no CMF value is greater than the random value

def drawSamples(pmf: List[float], numRandomValues: int) -> List[str]:
    """Generate random values and find the closest CMF values and their bucket indices."""
    cmf = computeCmf(pmf)
    randomValues = [random.uniform(0, 1) for _ in range(numRandomValues)]
    closestCmfValues = []
    samples = []
    
    for randomValue in randomValues:
        closest_cmf, bucketIdx = findClosestCmf(cmf, randomValue)
        closestCmfValues.append(closest_cmf)
        samples.append(buckets[bucketIdx])
    return samples

# Define the buckets for each point in list `p`
bucketMapping = {
    'ABCD': 1,
    'EFGH': 2,
    'IJKL': 4,
    'MNOP': 6,
    'QRST': 11,
    'UVXYZ': 3
}

# Convert list `p` to the list of bucket names (keys) and list of their associated values
p = list(bucketMapping.values())
buckets = list(bucketMapping.keys())

# Compute PMF
pmf = computePmf(p)

# Number of random values to generate
numRandomValues = 6

# Generate random values and closest CMF values
samples = drawSamples(pmf, numRandomValues)

print(f"Sampled Buckets: {samples}")