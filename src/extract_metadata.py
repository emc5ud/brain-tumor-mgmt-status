import pydicom as dicom
import pandas as pd
import swifter
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

FIELDS = [
    'AccessionNumber',
    'AcquisitionMatrix',
    #'B1rms',                           # nan
    #'BitsAllocated',                   # const: 16
    #'BitsStored',                      # const: 16
    'Columns',
    'ConversionType',
    'DiffusionBValue',
    'DiffusionGradientOrientation',
    'EchoNumbers',
    #'EchoTime',                        # nan
    'EchoTrainLength',
    'FlipAngle',
    #'HighBit',                         # const: 15
    'HighRRValue',
    'ImageDimensions',
    'ImageFormat',
    'ImageGeometryType',
    'ImageLocation',
    'ImageOrientation',
    'ImageOrientationPatient',
    'ImagePosition',
    'ImagePositionPatient',
    #'ImageType',                       # [const: 'DERIVED', 'SECONDARY']            
    'ImagedNucleus',
    'ImagingFrequency',
    'InPlanePhaseEncodingDirection',
    'InStackPositionNumber',
    'InstanceNumber',
    #'InversionTime',                   # nan
    #'Laterality',                      # nan
    'LowRRValue',
    'MRAcquisitionType',
    'MagneticFieldStrength',
    #'Modality',                        # const: MR
    'NumberOfAverages',
    'NumberOfPhaseEncodingSteps',
    'PatientID',
    'PatientName',
    #'PatientPosition',                 # const: HFS
    'PercentPhaseFieldOfView',
    'PercentSampling',
    # 'PhotometricInterpretation',      # const: MONOCHROME2
    'PixelBandwidth',
    'PixelPaddingValue',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    #'PositionReferenceIndicator',      # nan
    'PresentationLUTShape',
    'ReconstructionDiameter',
    #'RescaleIntercept',                # const: 0.0
    #'RescaleSlope',                    # const: 1.0
    #'RescaleType',                     # const: US    'Rows',
    'SAR',
    #'SOPClassUID',                     # const: 1.2.840.10008.5.1.4.1.1.4
    'SOPInstanceUID',
    #'SamplesPerPixel',                 # const: 1
    'SeriesDescription',
    'SeriesInstanceUID',
    'SeriesNumber',
    'SliceLocation',
    'SliceThickness',
    'SpacingBetweenSlices',             # const: 1
    'SpatialResolution',
    'SpecificCharacterSet',
    'StudyInstanceUID',
    'TemporalResolution',
    #'TransferSyntaxUID',               # const: 1.2.840.10008.1.2
    'TriggerWindow',
    'WindowCenter',
    'WindowWidth'
]

SERIES_TYPES = ['FLAIR', 'T1wCE', 'T1w', 'T2w']


def get_dicom_files(labels):
    dicom_files = []
    for patient_id in labels.str_id:
        for file_type in SERIES_TYPES:
            file_path = f'data/train/{patient_id}/{file_type}/*.dcm'
            dicom_files += glob(file_path)
    return dicom_files


def get_meta_info(dicom_path):
    dicom_file = dicom.read_file(dicom_path, force=True)
    row = {f: dicom_file.get(f) for f in FIELDS}
    return row


def create_meta_df(dicom_files):
    with Pool(14) as p:
        meta_dicts = list(tqdm(p.imap(get_meta_info, dicom_files), total=len(dicom_files)))
    # convert list of dicts to pandas df
    return pd.json_normalize(meta_dicts)


def get_label_df(data_dir):
    labels = pd.read_csv(f'{data_dir}/train_labels.csv')
    labels['str_id'] = labels.BraTS21ID.astype(str).str.zfill(5)
    return labels


def main(data_dir='./data/'):
    labels = get_label_df(data_dir)
    dicom_files = get_dicom_files(labels)
    meta_df = create_meta_df(dicom_files)
    meta_df.to_csv(data_dir + 'train_metadata.csv', index=False)


if __name__ == "__main__":
    main()