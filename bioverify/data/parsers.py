"""
Dataset parsers for different biometric dataset structures.

Each parser understands a specific dataset's directory layout and file naming
conventions, extracting metadata and returning a list of image records.
"""
import os
import re
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ImageRecord:
    """Single image record with extracted metadata."""
    image_path: str
    identity: str
    modality: str
    dataset_name: str
    side: Optional[str] = None  # L/R for iris, hand
    finger: Optional[str] = None  # For finger vein
    modality_type: Optional[str] = None  # e.g., dorsal vs finger vein
    image_type: Optional[str] = None  # e.g., raw vs roi
    sample_id: Optional[str] = None
    session: Optional[str] = None
    angle: Optional[str] = None  # For face
    lighting: Optional[str] = None  # For face
    expression: Optional[str] = None  # For face
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetParser(ABC):
    """Base class for dataset parsers."""
    
    def __init__(self, dataset_root: str, dataset_name: str, modality: str, **kwargs):
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.modality = modality
        # kwargs ignored in base class but allows subclasses to accept additional parameters
        
    @abstractmethod
    def parse(self) -> List[ImageRecord]:
        """Parse dataset and return list of image records."""
        pass
    
    def _get_valid_images(self, directory: Path, extensions: tuple = ('.jpg', '.png', '.bmp')) -> List[Path]:
        """Recursively find all image files in directory, excluding __MACOSX metadata folders."""
        image_files = []
        for ext in extensions:
            image_files.extend(directory.rglob(f'*{ext}'))
        # Filter out macOS metadata files from __MACOSX folders
        image_files = [f for f in image_files if '__MACOSX' not in f.parts]
        return sorted(image_files)


class CASIAIrisParser(DatasetParser):
    """Parser for CASIA-Iris-Thousand dataset.
    
    Structure: identity/ -> L/R/ -> S{identity}{side}{sample}.jpg
    Example: 050/L/S5050L01.jpg
    """
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        # Find all identity folders (000-999)
        identity_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir() and d.name.isdigit()]
        
        for identity_dir in identity_dirs:
            identity = identity_dir.name
            
            # Check L and R subdirectories
            for side_dir in ['L', 'R']:
                side_path = identity_dir / side_dir
                if not side_path.exists():
                    continue
                    
                # Find all images in this side
                images = self._get_valid_images(side_path)
                
                for img_path in images:
                    # Extract sample number from filename (e.g., S5050L01.jpg -> 01)
                    match = re.search(r'[LR](\d+)\.', img_path.name)
                    sample_id = match.group(1) if match else None
                    
                    record = ImageRecord(
                        image_path=str(img_path),
                        identity=identity,
                        modality=self.modality,
                        dataset_name=self.dataset_name,
                        side=side_dir,
                        sample_id=sample_id
                    )
                    records.append(record)
        
        return records


class MMUIrisParser(DatasetParser):
    """Parser for MMU Iris Database.
    
    Structure: identity/ -> left/right/ -> *.bmp
    Example: similar to CASIA but with .bmp files and full word sides
    """
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        identity_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        
        for identity_dir in identity_dirs:
            identity = identity_dir.name
            
            # Check both 'left'/'right' and 'L'/'R' for compatibility
            for side_name, side_abbr in [('left', 'L'), ('right', 'R'), ('L', 'L'), ('R', 'R')]:
                side_path = identity_dir / side_name
                if not side_path.exists():
                    continue
                    
                images = self._get_valid_images(side_path, extensions=('.bmp',))
                
                for idx, img_path in enumerate(images):
                    record = ImageRecord(
                        image_path=str(img_path),
                        identity=identity,
                        modality=self.modality,
                        dataset_name=self.dataset_name,
                        side=side_abbr,
                        sample_id=str(idx)
                    )
                    records.append(record)
        
        return records


class AMFIrisParser(DatasetParser):
    """Parser for AMF Iris Dataset.
    
    Structure: identity/ -> *.bmp (flat structure, no L/R subdirs)
    """
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        identity_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        
        for identity_dir in identity_dirs:
            identity = identity_dir.name
            images = self._get_valid_images(identity_dir, extensions=('.bmp',))
            
            for idx, img_path in enumerate(images):
                # Parse filename to infer side: Iris_{date}_{time}_{side}.bmp
                side = img_path.stem.split('_')[-1] if '_' in img_path.stem else None
                record = ImageRecord(
                    image_path=str(img_path),
                    identity=identity,
                    modality=self.modality,
                    dataset_name=self.dataset_name,
                    side=side,
                    sample_id=str(idx)
                )
                records.append(record)
        
        return records


class MultiPIEParser(DatasetParser):
    """Parser for CMU Multi-PIE Face Dataset.
    
    Structure: Multi_Pie_{version}/ -> {identity}_{01}_{expression}_{angle}_{light}_crop_128.png
    Example: Multi_Pie_HR_128/001_01_01_051_05_crop_128.png
    
    Supports two versions:
    - HR_128: High resolution 128x128
    - LR_4x_Nearest: Low resolution 4x downsampled with nearest neighbor
    """
    
    def __init__(self, dataset_root: str, dataset_name: str, modality: str, image_type: str = 'HR_128', **kwargs):
        """
        Args:
            dataset_root: Path to dataset root directory
            dataset_name: Name of the dataset
            modality: Biometric modality
            image_type: Which version to parse - 'HR_128', 'LR_4x', or 'both'. Default: 'HR_128'
        """
        super().__init__(dataset_root, dataset_name, modality)
        self.image_type = image_type
        if self.image_type not in ['HR_128', 'LR_4x', 'both']:
            raise ValueError(f"image_type must be 'HR_128', 'LR_4x', or 'both', got: {image_type}")
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        # Determine which folders to process
        folders_to_process = []
        if self.image_type in ['HR_128', 'both']:
            folders_to_process.append(('Multi_Pie_HR_128', 'HR_128'))
        if self.image_type in ['LR_4x', 'both']:
            folders_to_process.append(('Multi_Pie_LR_4x_Nearest', 'LR_4x'))
        
        for folder_name, version in folders_to_process:
            version_path = self.dataset_root / folder_name
            if not version_path.exists():
                continue
            
            images = self._get_valid_images(version_path, extensions=('.png',))
            
            for img_path in images:
                # Parse filename: {identity}_{01}_{expression}_{angle}_{light}_crop_128.png
                parts = img_path.stem.split('_')
                if len(parts) >= 5:
                    identity = parts[0]
                    expression = parts[2]
                    angle = parts[3]
                    lighting = parts[4]
                    
                    record = ImageRecord(
                        image_path=str(img_path),
                        identity=identity,
                        modality=self.modality,
                        dataset_name=self.dataset_name,
                        image_type=version,
                        expression=expression,
                        angle=angle,
                        lighting=lighting,
                        metadata={'version': version,
                                  'angle': angle,
                                  'lighting': lighting}
                    )
                    records.append(record)
        
        return records


class _11kHandsParser(DatasetParser):
    """Parser for 11k Hands dataset.
    
    Structure: Hands/ -> *.jpg
    There is a Handinfo.csv file with metadata.
    its structure is as follows: id,age,gender,skinColor,accessories,nailPolish,aspectOfHand,imageName,irregularities 
    """
    
    def parse(self) -> List[ImageRecord]:
        records = []

        handInfo_path = self.dataset_root / 'HandInfo.csv'
        handsDir = self.dataset_root / 'Hands'

        images = self._get_valid_images(handsDir, extensions=('.jpg',))
        
        # Read CSV data if it exists
        csv_data = []
        if handInfo_path.exists():
            with open(handInfo_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                csv_data = list(reader)
        
        # Iterate through images and assign metadata from CSV in order
        for idx, img_path in enumerate(images):
            filename = img_path.stem
            
            # Get metadata from CSV if available
            metadata = {'filename': filename}
            identity = filename  # Default identity is filename
            
            if idx < len(csv_data):
                row = csv_data[idx]
                identity = row.get('id', filename)
                metadata.update({
                    'age': row.get('age'),
                    'gender': row.get('gender'),
                    'skinColor': row.get('skinColor'),
                    'accessories': row.get('accessories'),
                    'nailPolish': row.get('nailPolish'),
                    'aspectOfHand': row.get('aspectOfHand'),
                    'imageName': row.get('imageName'),
                    'irregularities': row.get('irregularities')
                })
            
            record = ImageRecord(
                image_path=str(img_path),
                identity=identity,
                modality=self.modality,
                dataset_name=self.dataset_name,
                metadata=metadata
            )
            records.append(record)
        
        return records


class FingerVeinParser(DatasetParser):
    """Parser for Finger Vein Database.
    
    Structure: Identity/ -> L/R (ruka) -> {index,middle,ring}_{1-6}.bmp
    Example: 001/L/index_1.bmp, 002/R/middle_3.bmp
    """
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        # Find all identity folders
        identity_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        
        for identity_dir in identity_dirs:
            identity = identity_dir.name
            
            # Check left and right subdirectories
            for side_dir in ['left', 'right']:
                side_path = identity_dir / side_dir
                if not side_path.exists():
                    continue
                    
                # Find all images in this side
                images = self._get_valid_images(side_path, extensions=('.bmp',))
                
                for img_path in images:
                    # Parse filename: {finger}_{sample}.bmp
                    # e.g., index_1.bmp, middle_3.bmp, ring_6.bmp
                    match = re.match(r'(index|middle|ring)_(\d+)', img_path.stem)
                    
                    if match:
                        finger = match.group(1)
                        sample_id = match.group(2)
                    else:
                        finger = None
                        sample_id = None
                    
                    record = ImageRecord(
                        image_path=str(img_path),
                        identity=identity,
                        modality=self.modality,
                        dataset_name=self.dataset_name,
                        side=side_dir,
                        finger=finger,
                        sample_id=sample_id
                    )
                    records.append(record)
        
        return records


class EEMSCDBMParser(DatasetParser):
    """Parser for EEMSC-DBM Finger Vein Database.
    
    Structure: data -> Identity (0001-0060) -> {id}_{finger}_{sample}_{date}.png
    Example: data/0001/0001_1_01_20150101.png
    
    NOTE: Finger mapping unknown - fingers are numbered 1-6 but the actual
    correspondence to physical fingers is not documented in the dataset.
    The finger number is saved as-is in the finger field for future reference
    if the mapping becomes available.
    """
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        data_path = self.dataset_root / 'data'
        if not data_path.exists():
            return records
        
        # Find all identity folders
        identity_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        for identity_dir in sorted(identity_dirs):
            identity = identity_dir.name
            
            # Find all images in identity folder
            images = self._get_valid_images(identity_dir, extensions=('.png',))
            
            for img_path in images:
                # Parse filename: {id}_{finger}_{sample}_{date}.png
                # e.g., 0001_1_01_20150101.png
                match = re.match(r'(\d+)_(\d)_(\d+)(?:_\d+)?', img_path.stem)
                
                if match:
                    file_id = match.group(1)
                    finger = match.group(2)  # 1-6, mapping unknown
                    sample_id = match.group(3)
                else:
                    finger = None
                    sample_id = None
                
                record = ImageRecord(
                    image_path=str(img_path),
                    identity=identity,
                    modality=self.modality,
                    dataset_name=self.dataset_name,
                    finger=finger,
                    sample_id=sample_id,
                    metadata={
                    }
                )
                records.append(record)
        
        return records


class THUFVFDTParser(DatasetParser):
    """Parser for THU-FVFDT dataset (Tsinghua Finger Vein and Finger Dorsal Vein Database).
    
    Structure: {FDT,FV}{1,2} -> {FDT,FV}{1,2}_{Train,Test} -> Identity -> 1.bmp
    Where:
        FDT = Dorsal vein (palm)
        FV = Finger Vein
        1 = Raw
        2 = ROI
    
    Example: FDT1/FDT1_Train/001/1.bmp (dorsal, raw, train session, identity 001)
    """
    
    def __init__(self, dataset_root: str, dataset_name: str, modality: str, image_type: str = 'roi', modality_type: str = 'vein'):
        """
        Args:
            dataset_root: Path to dataset root directory
            dataset_name: Name of the dataset
            modality: Biometric modality
            image_type: Which images to parse - 'raw', 'roi', or 'both'. Default: 'roi'
                - raw: FV1/FDT1 folders
                - roi: FV2/FDT2 folders
                - both: all folders
            modality_type: Type of modality - 'vein', 'dorsal', or 'both'. Default: 'vein'
                - vein: FV1/FV2 (finger vein)
                - dorsal: FDT1/FDT2 (dorsal vein)
                - both: all folders
        """
        super().__init__(dataset_root, dataset_name, modality)
        self.image_type = image_type.lower()
        if self.image_type not in ['raw', 'roi', 'both']:
            raise ValueError(f"image_type must be 'raw', 'roi', or 'both', got: {image_type}")
        self.modality_type = modality_type.lower()
        if self.modality_type not in ['vein', 'dorsal', 'both']:
            raise ValueError(f"modality_type must be 'vein', 'dorsal', or 'both', got: {modality_type}")
        
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        # Determine which folder prefixes to process based on modality_type
        prefixes_to_process = []
        if self.modality_type in ['vein', 'both']:
            prefixes_to_process.append('FV')
        if self.modality_type in ['dorsal', 'both']:
            prefixes_to_process.append('FDT')
        
        # Determine which codes to process based on image_type
        codes_to_process = []
        if self.image_type in ['raw', 'both']:
            codes_to_process.append('1')
        if self.image_type in ['roi', 'both']:
            codes_to_process.append('2')
        
        # Find all modality type folders: FDT1, FDT2, FV1, FV2
        modality_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        
        for modality_dir in modality_dirs:
            folder_name = modality_dir.name
            
            # Parse folder name: {FDT,FV}{1,2}
            match = re.match(r'(FDT|FV)([12])', folder_name)
            if not match:
                continue
            
            prefix = match.group(1)
            code = match.group(2)
            
            # Check if this folder should be processed
            if prefix not in prefixes_to_process or code not in codes_to_process:
                continue
            
            modality_type_str = 'dorsal' if prefix == 'FDT' else 'vein'
            img_type = 'raw' if code == '1' else 'roi'
            
            # Find session folders: FDT1_Train, FDT1_Test, etc.
            session_dirs = [d for d in modality_dir.iterdir() if d.is_dir()]
            
            for session_dir in session_dirs:
                session_name = session_dir.name
                
                # Extract session type from folder name (Train or Test)
                if 'Train' in session_name or 'train' in session_name:
                    session = 'train'
                elif 'Test' in session_name or 'test' in session_name:
                    session = 'test'
                else:
                    session = None
                
                # Find identity folders
                identity_dirs = [d for d in session_dir.iterdir() if d.is_dir()]
                
                for identity_dir in identity_dirs:
                    identity = identity_dir.name
                    
                    # Find all images in identity folder
                    images = self._get_valid_images(identity_dir, extensions=('.bmp',))
                    
                    for img_path in images:
                        sample_id = img_path.stem
                        
                        record = ImageRecord(
                            image_path=str(img_path),
                            identity=identity,
                            modality=self.modality,
                            dataset_name=self.dataset_name,
                            modality_type=modality_type_str,
                            image_type=img_type,
                            metadata={
                                'session': session
                            }
                        )
                        records.append(record)
        
        return records


class FVUSMParser(DatasetParser):
    """Parser for FV-USM Finger Vein Database.
    
    Structure: 1st_session/2nd_session -> extractedvein/raw_data -> vein{id}_{type} -> {01-06}.jpg
    Where type encodes side and finger:
        1 - left index
        2 - left middle
        3 - right index
        4 - right middle
    
    Example: 1st_session/extractedvein/vein0001_1/01.jpg (subject 1, left index, 1st session)
    
    Note: This dataset has both raw images (raw_data) and extracted vein images (extractedvein).
    Use image_type parameter to specify which to use: 'raw', 'roi', or 'both'.
    """
    
    def __init__(self, dataset_root: str, dataset_name: str, modality: str, image_type: str = 'roi', **kwargs):
        """
        Args:
            dataset_root: Path to dataset root directory
            dataset_name: Name of the dataset
            modality: Biometric modality
            image_type: Which images to parse - 'raw', 'roi', or 'both'. Default: 'roi'
        """
        super().__init__(dataset_root, dataset_name, modality)
        self.image_type = image_type.lower()
        if self.image_type not in ['raw', 'roi', 'both']:
            raise ValueError(f"image_type must be 'raw', 'roi', or 'both', got: {image_type}")
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        # Mapping of type suffix to side and finger
        type_mapping = {
            '1': ('L', 'index'),
            '2': ('L', 'middle'),
            '3': ('R', 'index'),
            '4': ('R', 'middle'),
        }
        
        # Determine which folders to process
        folders_to_process = []
        if self.image_type in ['raw', 'both']:
            folders_to_process.append(('raw_data', 'raw'))
        if self.image_type in ['roi', 'both']:
            folders_to_process.append(('extractedvein', 'roi'))
        
        # Find all session folders (1st_session, 2nd_session, etc.)
        session_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        
        for session_dir in session_dirs:
            session_name = session_dir.name
            
            # Check specified image source folders
            for folder_name, img_type in folders_to_process:
                source_path = session_dir / folder_name
                if not source_path.exists():
                    continue
                
                # Find all vein subject folders: vein{id}_{type}
                vein_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.startswith('vein')]
                
                for vein_dir in vein_dirs:
                    # Parse directory name: vein{id}_{type}
                    # e.g., vein0001_1, vein0102_3
                    match = re.match(r'vein(\d+)_([1234])', vein_dir.name)
                    
                    if match:
                        identity = match.group(1)
                        type_code = match.group(2)
                        side, finger = type_mapping.get(type_code, (None, None))
                    else:
                        continue
                    
                    # Find all images in this vein directory
                    images = self._get_valid_images(vein_dir, extensions=('.jpg',))
                    
                    for img_path in images:
                        # Extract sample number from filename: 01.jpg -> 01
                        sample_id = img_path.stem
                        
                        record = ImageRecord(
                            image_path=str(img_path),
                            identity=identity,
                            modality=self.modality,
                            image_type=img_type,
                            dataset_name=self.dataset_name,
                            side=side,
                            finger=finger,
                            sample_id=sample_id,
                            metadata={
                                'session': session_name,
                                'image_type': img_type
                            }
                        )
                        records.append(record)
        
        return records


class MMCBNUParser(DatasetParser):
    """Parser for MMCBNU_6000 Finger Vein Database.
    
    Structure: Captured images/ROIs -> Identity -> {L,R}_{Fore,Middle,Ring} -> {01-10}.bmp
    Example: Captured images/001/L_Fore/01.bmp or ROIs/001/R_Middle/05.bmp
    
    Note: This dataset has both raw images (Captured images) and ROI images (ROIs).
    Use image_type parameter to specify which to use: 'raw', 'roi', or 'both'.
    """
    
    def __init__(self, dataset_root: str, dataset_name: str, modality: str, image_type: str = 'roi', **kwargs):
        """
        Args:
            dataset_root: Path to dataset root directory
            dataset_name: Name of the dataset
            modality: Biometric modality
            image_type: Which images to parse - 'raw', 'roi', or 'both'. Default: 'roi'
        """
        super().__init__(dataset_root, dataset_name, modality)
        self.image_type = image_type.lower()
        if self.image_type not in ['raw', 'roi', 'both']:
            raise ValueError(f"image_type must be 'raw', 'roi', or 'both', got: {image_type}")
    
    def parse(self) -> List[ImageRecord]:
        records = []
        
        # Determine which folders to process
        folders_to_process = []
        if self.image_type in ['raw', 'both']:
            folders_to_process.append(('Captured images', 'raw'))
        if self.image_type in ['roi', 'both']:
            folders_to_process.append(('ROIs', 'roi'))
        
        for folder_name, img_type in folders_to_process:
            base_path = self.dataset_root / folder_name
            if not base_path.exists():
                continue
            
            # Find all identity folders
            identity_dirs = [d for d in base_path.iterdir() if d.is_dir()]
            
            for identity_dir in identity_dirs:
                identity = identity_dir.name
                
                # Find all finger subdirectories: {L,R}_{Fore,Middle,Ring}
                finger_dirs = [d for d in identity_dir.iterdir() if d.is_dir()]
                
                for finger_dir in finger_dirs:
                    # Parse directory name: L_Fore, R_Middle, etc.
                    match = re.match(r'([LR])_(Fore|Middle|Ring)', finger_dir.name)
                    
                    if match:
                        side = match.group(1)
                        finger = match.group(2).lower()
                    else:
                        side = None
                        finger = None
                    
                    # Find all images in this finger directory
                    images = self._get_valid_images(finger_dir, extensions=('.bmp',))
                    
                    for img_path in images:
                        # Extract sample number from filename: 01.bmp -> 01
                        sample_id = img_path.stem
                        
                        record = ImageRecord(
                            image_path=str(img_path),
                            identity=identity,
                            modality=self.modality,
                            dataset_name=self.dataset_name,
                            side=side,
                            finger=finger,
                            sample_id=sample_id,
                            metadata={'image_type': img_type}
                        )
                        records.append(record)
        
        return records


# Registry of parsers by dataset pattern matching
PARSER_REGISTRY = {
    'CASIA-Iris-Thousand': CASIAIrisParser,
    'MMU-Iris': MMUIrisParser,
    'AMF-Iris': AMFIrisParser,
    'Multi-PIE': MultiPIEParser,
    'Multi_PIE': MultiPIEParser,
    '11k-Hands': _11kHandsParser,
    'Hands': _11kHandsParser,
    'Finger-Vein': FingerVeinParser,
    'FingerVein': FingerVeinParser,
    'FV-USM': FVUSMParser,
    'FVUSM': FVUSMParser,
    'EEMSC-DBM': EEMSCDBMParser,
    'EEMSCDBM': EEMSCDBMParser,
    'THU-FVFDT': THUFVFDTParser,
    'THU_FVFDT': THUFVFDTParser,
    'MMCBNU': MMCBNUParser,
    'MMCBNU_6000': MMCBNUParser,
}


def get_parser(dataset_path: str, dataset_name: str, modality: str, **kwargs) -> DatasetParser:
    """Factory function to get appropriate parser for a dataset.
    
    Args:
        dataset_path: Path to dataset root directory
        dataset_name: Name/identifier of the dataset
        modality: Biometric modality (iris, face, hand, fingervein)
        **kwargs: Additional parameters for specific parsers (e.g., image_type='roi' for MMCBNU)
        
    Returns:
        Appropriate DatasetParser subclass instance
        
    Raises:
        ValueError: If no suitable parser found
    """
    # Try exact match first
    for key, parser_class in PARSER_REGISTRY.items():
        if key.lower() in dataset_name.lower():
            return parser_class(dataset_path, dataset_name, modality, **kwargs)
    
    raise ValueError(f"No suitable parser found for dataset: {dataset_name} (modality: {modality})")
