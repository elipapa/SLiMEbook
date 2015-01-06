Each benchmark folder contains four files. All files are formatted for easy loading into the R statistical programming environment. Files are tab-delimited. There are column headers and row labels, except that there is no header for the first column (the column of row labels), in keeping with R usage.

For example, in the FS folder:

FS_labels.txt - this contains the true class label for each sample. One sample per row. The first column contains the names of the samples (row labels).

FS_otus.txt - this contains the OTU relative abundance table. One sample per row. The first column contains the names of the samples (row labels). Each other column represents one OTU. The column headers in the first row are the unique ID's of the OTUs.

FS_otu_lineages.txt - this contains the RDP-based taxonomy assignments for the OTUs. Each row represents one OTU. The first column contains the OTU ID, the second column the RDP lineage.

FS_test_indices.txt - a matrix containing the train/test split assignments for the 10 random train/test splits used in the paper. Each row corresponds to a sample; each column corresponds to a different randomly chosen test set. A value of "TRUE" means that that sample belongs in the test set for that train/test split.