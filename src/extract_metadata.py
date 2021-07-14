import pydicom as dicom
import pandas as pd
import swifter
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

FIELDS = [
    'AccessionNumber',
    'AcquisitionMatrix',
    'B1rms',
    'BitsAllocated',
    'BitsStored',
    'Columns',
    'ConversionType',
    'DiffusionBValue',
    'DiffusionGradientOrientation',
    'EchoNumbers',
    'EchoTime',
    'EchoTrainLength',
    'FlipAngle',
    'HighBit',
    'HighRRValue',
    'ImageDimensions',
    'ImageFormat',
    'ImageGeometryType',
    'ImageLocation',
    'ImageOrientation',
    'ImageOrientationPatient',
    'ImagePosition',
    'ImagePositionPatient',
    'ImageType',
    'ImagedNucleus',
    'ImagingFrequency',
    'InPlanePhaseEncodingDirection',
    'InStackPositionNumber',
    'InstanceNumber',
    'InversionTime',
    'Laterality',
    'LowRRValue',
    'MRAcquisitionType',
    'MagneticFieldStrength',
    'Modality',
    'NumberOfAverages',
    'NumberOfPhaseEncodingSteps',
    'PatientID',
    'PatientName',
    'PatientPosition',
    'PercentPhaseFieldOfView',
    'PercentSampling',
    'PhotometricInterpretation',
    'PixelBandwidth',
    'PixelPaddingValue',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    'PositionReferenceIndicator',
    'PresentationLUTShape',
    'ReconstructionDiameter',
    'RescaleIntercept',
    'RescaleSlope',
    'RescaleType',
    'Rows',
    'SAR',
    'SOPClassUID',
    'SOPInstanceUID',
    'SamplesPerPixel',
    'SeriesDescription',
    'SeriesInstanceUID',
    'SeriesNumber',
    'SliceLocation',
    'SliceThickness',
    'SpacingBetweenSlices',
    'SpatialResolution',
    'SpecificCharacterSet',
    'StudyInstanceUID',
    'TemporalResolution',
    'TransferSyntaxUID',
    'TriggerWindow',
    'WindowCenter',
    'WindowWidth'
]

FM_FIELDS = [
    'FileMetaInformationGroupLength',
    'FileMetaInformationVersion',
    'ImplementationClassUID',
    'ImplementationVersionName',
    'MediaStorageSOPClassUID',
    'MediaStorageSOPInstanceUID',
    'SourceApplicationEntityTitle',
    'TransferSyntaxUID',
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
    row_fm = {f: dicom_file.file_meta.get(f) for f in FM_FIELDS}
    row_other = {
        'is_original_encoding': dicom_file.is_original_encoding,
        'is_implicit_VR': dicom_file.is_implicit_VR,
        'is_little_endian': dicom_file.is_little_endian,
        'timestamp': dicom_file.timestamp,
    }
    return {**row, **row_fm, **row_other}


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
    meta_df.to_csv(data_dir + 'train_metadata.csv')


if __name__ == "__main__":
    main()