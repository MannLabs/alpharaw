# Resources Directory

## psi-ms.owl

### Overview
The `psi-ms.owl` file contains the Proteomics Standards Initiative Mass Spectrometry (PSI-MS) Controlled Vocabulary (CV) in OWL format. This ontology defines standardized terms for describing mass spectrometry data and metadata, ensuring semantic consistency and interoperability in proteomics data exchange.

### Purpose
This file is used by the mzML writer to:
- Validate and retrieve official ontology term names and accession numbers
- Ensure mzML output complies with HUPO-PSI standards
- Provide semantic meaning to mass spectrometry data elements through controlled vocabulary terms
- Map internal data representations to standardized PSI-MS CV terms

### Source Information
- **Official Website**: https://www.psidev.info/mzml
- **Source Repository**: https://github.com/HUPO-PSI/psi-ms-CV
- **Current Version**: 4.1.195
- **Download URL**: https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.owl
- **Added to alpharaw**: Commit `55fb572` (July 2, 2025)
- **Moved to resources**: Commit `364391d` (November 4, 2025)

### Updating the File
To update to the latest version of the PSI-MS CV ontology:

```bash
cd alpharaw/resources
curl -O https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.owl
```

After updating, verify the new version by checking the `owl:versionIRI` in the file header.

### License
The PSI-MS Controlled Vocabulary is released under the Creative Commons CC0 license.
See: https://github.com/HUPO-PSI/psi-ms-CV/blob/master/Licence.txt
