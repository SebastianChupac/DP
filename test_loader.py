"""Simple test script to verify PairDataset loading."""
from bioverify.data.dataset import PairDataset

# Load training pairs
ds = PairDataset(
    'bioverify/data/pairs/test001_finger_thufvfdt.csv', 
    base_path='C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset'
)

print(f'Loaded {len(ds)} training pairs')

# Get first pair
pair = ds[0]

print(f'\nFirst pair:')
print(f'  ID: {pair["pair_id"]}')
print(f'  Image 1: ...{pair["image1_path"][-60:]}')
print(f'  Image 2: ...{pair["image2_path"][-60:]}')
print(f'  Same person: {pair["ground_truth"]}')
print(f'  Identities: {pair["identity1"]} vs {pair["identity2"]}')
print(f'  Modality: {pair["modality"]}')

# Show statistics
print('\n' + '='*50)
ds.print_statistics()
