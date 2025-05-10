#!/usr/bin/env python3
"""
Chromophore CSV Preprocessing Module

Processes the raw DB_for_chromophore.csv to produce stratified train/test splits
for absorption wavelength modeling.

Steps:
1. Load raw CSV from config.RAW_CSV
2. Canonicalize SMILES
3. Generate Morgan fingerprints (radius, bits from config)
4. Convert Absorption max (nm) to energy (eV)
5. Flag outliers by IQR on nm and eV
6. Write full, train, and test CSVs to config-defined paths
"""

import os
import logging
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split

import config
from config import OUTLIER_MULTIPLIER


class ChromophoreProcessor:
    def __init__(
        self,
        csv_path: str = config.RAW_CSV,
        output_dir: str = config.PROCESSED_DATA_DIR,
        fingerprint_radius: int = getattr(config, 'FP_RADIUS', 2),
        fingerprint_size: int = getattr(config, 'FP_SIZE', 1024),
        outlier_multiplier: float = getattr(config, 'OUTLIER_MULTIPLIER', 1.5),
        test_fraction: float = getattr(config, 'TEST_SIZE', 0.2),
        random_seed: int = getattr(config, 'RANDOM_SEED', 42),
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.fpradius = fingerprint_radius
        self.fpsize = fingerprint_size
        self.iqr_k = outlier_multiplier
        self.test_frac = test_fraction
        self.seed = random_seed

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        logging.info("Loading raw CSV: %s", self.csv_path)
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"Raw CSV not found at {self.csv_path}")
        return pd.read_csv(self.csv_path)

    @staticmethod
    def canonicalize(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

    def fingerprint(self, smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fpradius, fpSize=self.fpsize
        )
        return gen.GetFingerprint(mol).ToBitString()

    @staticmethod
    def to_ev(nm: float) -> Optional[float]:
        return 1240.0 / nm if nm and nm > 0 else None

    def flag_outliers(self, series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
        return (series < lower) | (series > upper)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            tag = row.get('Tag') or row.get('ID')
            raw_smiles = row.get('Chromophore')
            abs_nm = row.get('Absorption max (nm)')
            if pd.isna(raw_smiles) or pd.isna(abs_nm):
                logging.debug("Skipping %s: missing data", tag)
                continue
            can = self.canonicalize(raw_smiles)
            if not can:
                logging.warning("Invalid SMILES for %s", tag)
                continue
            fp = self.fingerprint(can)
            if not fp:
                logging.warning("Fingerprint failure for %s", tag)
                continue
            ev = self.to_ev(abs_nm)
            records.append({
                'Tag': tag,
                'Canonical_SMILES': can,
                'Fingerprint': fp,
                'Absorption max (nm)': abs_nm,
                'Energy_eV': ev,
            })
        dfp = pd.DataFrame(records)
        if not dfp.empty:
            dfp['Outlier_nm'] = self.flag_outliers(dfp['Absorption max (nm)'])
            dfp['Outlier_ev'] = self.flag_outliers(dfp['Energy_eV'])
            dfp['Is_Outlier'] = dfp['Outlier_nm'] | dfp['Outlier_ev']
        return dfp

    def split_and_export(self, dfp: pd.DataFrame) -> None:
        full_path = config.PROCESSED_CSV
        train_path = config.TRAIN_CSV
        test_path = config.TEST_CSV

        dfp.to_csv(full_path, index=False)
        ids = dfp['Tag'].unique()
        strat = dfp.groupby('Tag')['Is_Outlier'].any().reindex(ids).fillna(False)
        train_ids, test_ids = train_test_split(
            ids,
            test_size=self.test_frac,
            random_state=self.seed,
            stratify=strat,
        )
        dfp[dfp['Tag'].isin(train_ids)].to_csv(train_path, index=False)
        dfp[dfp['Tag'].isin(test_ids)].to_csv(test_path, index=False)

        logging.info("Saved full data to %s", full_path)
        logging.info("Saved train split to %s", train_path)
        logging.info("Saved test split to %s", test_path)


def main():
    processor = ChromophoreProcessor()
    df_raw = processor.load_data()
    logging.info("Raw entries: %d", len(df_raw))
    df_proc = processor.preprocess(df_raw)
    logging.info("Processed entries: %d", len(df_proc))
    processor.split_and_export(df_proc)


if __name__ == '__main__':
    main()
