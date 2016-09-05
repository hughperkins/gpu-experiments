// this assumes that all matrices are column major

kernel void {{kernelname}} (
        int GlobalRows, int GlobalMids, int GlobalCols,
        int BlockRows, int BlockMids, int BlockCols,
        int blockRows, int blockMids, int blockCols,
        global float4 *C_float4, global float4 *A_float4, global float4 *B_float4,
        local float4 *B_block_float4, local float *A_block_unused) {
    global float *C = (global float *)C_float4;
    global float *B = (global float *)B_float4;
    global float *A = (global float *)A_float4;
    local float *B_block = (local float *)B_block_float4;
    int BlockRow = get_group_id(0);
    int BlockCol = get_group_id(1);
    int tid = get_local_id(0);
    int globalRow = BlockRow * blockRows + tid;

    float C_row[{{blockCols}}];
    //float *C_row = (float *)C_row_float4;
    for(int blockCol=0; blockCol < {{blockCols}}; blockCol++) {
         C_row[blockCol] = 0.0f;
    }
    {
        for(int BlockMid = 0; BlockMid < BlockMids; BlockMid++) {
            // first copy down the data from B
            // each thread will handle one column of B data, ie
            // iterate over blockMid
            // sync point (can remove if num threads == warpsize)
            //barrier(CLK_LOCAL_MEM_FENCE);
            {
                int blockCol = tid;
                int globalCol = BlockCol * blockCols + blockCol;
                int BlockMid_blockMids = BlockMid * blockMids;
                int globalCol_GlobalMids = globalCol * GlobalMids;
                int BlockMid_blockMids_plus_globalCol_GlobalMids = BlockMid_blockMids + globalCol_GlobalMids;
                int blockCol_blockMids = blockCol * blockMids;
                #pragma unroll 2
                for(int blockMid=0; blockMid < {{blockMids}}; blockMid++) {
                    B_block[blockCol_blockMids + blockMid] = B[BlockMid_blockMids_plus_globalCol_GlobalMids + blockMid];
                }
            }

            float A_row[{{blockMids}}];
            //float *A_row = (float*)A_row_float4;
            {
                int BlockMid_blockMids = BlockMid * blockMids;
                //#pragma unroll 4
                for(int blockMid=0; blockMid < blockMids; blockMid++) {
                    int globalMid = BlockMid_blockMids + blockMid;
                    A_row[blockMid] = A[(globalMid << 10) + globalRow];
                }
            }

            // sync point (can remove if num threads == warpsize)
            //barrier(CLK_LOCAL_MEM_FENCE);

            // calc some C :-)
            // each thread handles a row of c, so needs to iterate over columns
            // but for each column, needs to iterate over middle too
            for(int blockMid=0; blockMid < {{blockMids}}; blockMid++) {
                for(int blockCol=0; blockCol < {{blockCols}}; blockCol++) {
                    C_row[blockCol] += A_row[blockMid] * B_block[blockCol * blockMids + blockMid];
                }
            }
        }
    }
    // write C out
    #pragma unroll
    for(int blockCol=0; blockCol < {{blockCols}}; blockCol++) {
        int globalCol = BlockCol * blockCols + blockCol;
        C[globalCol * GlobalRows + globalRow] = C_row[blockCol];
        //C_float4[globalRow_globalCols_plus_BlockCol_blockCols + blockCol] = C_row_float4[blockCol];
    }
}
