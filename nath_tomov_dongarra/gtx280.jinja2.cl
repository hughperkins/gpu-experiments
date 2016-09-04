kernel void {{kernelname}} (
        int GlobalRows, int GlobalMids, int GlobalCols,
        int BlockRows, int BlockMids, int BlockCols,
        int blockRows, int blockMids, int blockCols,
        global float *C, global float *A, global float *B,
        local float *B_block) {
    int BlockRow = get_group_id(0);
    int BlockCol = get_group_id(1);
    int tid = get_local_id(0);
    // int blockRow = get_local_id(0);
    int globalRow = BlockRow * blockRows + tid;

    float C_row[{{blockCols}}];
    //float C_row1[{{blockCols}} >> 1];
    for(int blockCol=0; blockCol < blockCols; blockCol++) {
         C_row[blockCol] = 0.0f;
      //  C_row1[blockCol] = 0.0f;
    }
    {
        int blockCol = tid;
        int globalCol = BlockCol * blockCols + blockCol;
        for(int BlockMid = 0; BlockMid < BlockMids; BlockMid++) {
            // first copy down the data from B
            // each thread will handle one column of B data, ie
            // iterate over blockMid
            // sync point (can remove if num threads == warpsize)
            //barrier(CLK_LOCAL_MEM_FENCE);
            for(int blockMid=0; blockMid < blockMids; blockMid++) {
                int globalMid = BlockMid * blockMids + blockMid;
                B_block[blockMid * blockCols + blockCol] = B[globalMid * GlobalCols + globalCol];
            }
            // should probably copy down A too?  (otherwise have to wait for each float of A to come down,
            // one by one...)
            // but lets copy to private for now, no coasllescing, then try coallescing in v0.2
            float A_row[{{blockMids}}];
            for(int blockMid=0; blockMid < blockMids; blockMid++) {
                int globalMid = BlockMid * blockMids + blockMid;
                A_row[blockMid] = A[globalRow * GlobalMids + globalMid];
            }
            // sync point (can remove if num threads == warpsize)
            //barrier(CLK_LOCAL_MEM_FENCE);

            // calc some C :-)
            // each thread handles a row of c, so needs to iterate over columns
            // but for each column, needs to iterate over middle too
                for(int blockCol=0; blockCol < blockCols; blockCol++) {
            for(int blockMid=0; blockMid < blockMids; blockMid++) {
                int blockMid_blockCols = blockMid * blockCols;
                float a = A_row[blockMid];
                    C_row[blockCol] += a * B_block[blockMid_blockCols + blockCol];
                    // C_row1[blockCol] += A_row[blockMid + 1] * B_block[blockMid * blockCols + blockCol];
                }
            }
        }
    }
    // write C out
    int globalRow_globalCols = globalRow * GlobalCols;
    int BlockCol_blockCols = BlockCol * blockCols;
    int globalRow_globalCols_plus_BlockCol_blockCols = globalRow_globalCols + BlockCol_blockCols;
    for(int blockCol=0; blockCol < blockCols; blockCol++) {
        //int globalCol = BlockCol_blockCols + blockCol;
        C[globalRow_globalCols_plus_BlockCol_blockCols + blockCol] = C_row[blockCol];
    }
}
