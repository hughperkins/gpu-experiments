// this assumes that all matrices are column major

kernel void {{kernelname}} (
        int GlobalRows, int GlobalMids, int GlobalCols,
        int BlockRows, int BlockMids, int BlockCols,
        int blockRows, int blockMids, int blockCols,
        global float4 *C_float4, global float4 *A_float4, global float4 *B_float4,
        local float4 *B_block_float4, local float *A_block) {
    global float *C = (global float *)C_float4;
    global float *B = (global float *)B_float4;
    global float *A = (global float *)A_float4;
    local float *B_block = (local float *)B_block_float4;
    int BlockRow = get_group_id(0);
    int BlockCol = get_group_id(1);
    int tid = get_local_id(0);
    int globalRow = BlockRow * blockRows + tid;

    float4 C_row_float4[{{blockCols // 4}}];
    float *C_row = (float *)C_row_float4;
    for(int blockCol=0; blockCol < {{blockCols}}; blockCol++) {
         C_row[blockCol] = 0.0f;
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
            //#pragma unroll 4
            {
                int bcpy_midoffset = tid >> 3;
                int bcpy_col = tid & 7;
                int bcpy_globalCol = (BlockCol * blockCols >> 2) + bcpy_col;
                int BlockMid_blockMids = BlockMid * blockMids;
                for(int blockMid4=0; blockMid4 < blockMids; blockMid4 += 4) {
                    int blockMid = blockMid4 + bcpy_midoffset;
                    int globalMid = BlockMid_blockMids + blockMid;
                 //   B_block_float4[(blockMid * blockCols >> 2) + bcpy_col] = B_float4[(globalMid * GlobalCols >> 2) + bcpy_globalCol];
                }
            }

            // should probably copy down A too?  (otherwise have to wait for each float of A to come down,
            // one by one...)
            // but lets copy to private for now, no coasllescing, then try coallescing in v0.2
            float4 A_row_float4[{{blockMids // 4}}];
            float *A_row = (float*)A_row_float4;
            {
            int globalOffset = ((globalRow * GlobalMids) >> 2) + ((BlockMid * blockMids) >> 2);
            //#pragma unroll 2
            for(int blockMid=0; blockMid < {{blockMids // 4}}; blockMid++) {
               // A_row_float4[blockMid] = A_float4[globalOffset + blockMid];
            }
            }

            // sync point (can remove if num threads == warpsize)
            //barrier(CLK_LOCAL_MEM_FENCE);

            // calc some C :-)
            // each thread handles a row of c, so needs to iterate over columns
            // but for each column, needs to iterate over middle too
            for(int blockMid=0; blockMid < {{blockMids}}; blockMid++) {
                for(int blockCol=0; blockCol < {{blockCols}}; blockCol++) {
                    //C_row[blockCol] += A_row[blockMid] * B_block[blockCol * blockRows + blockMid];
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
