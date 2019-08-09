using UnityEngine;

namespace GPGPUExamples
{
    public class Reduction : MonoBehaviour
    {
        enum ExecMode
        {
            GPU,
            CPU,
        }

        [SerializeField] ComputeShader _computeShader;
        [SerializeField] int _size = 2147483;
        [SerializeField] ExecMode _mode = ExecMode.GPU;

        string[] kernelNames = 
        {
            "CsSetData", 
            "CsReduction",
        };

        uint _numOfGpuThreads; // number of threads per group
        int _numOfGroups;

        float[] _dataArray;    
        float[] _partialSums;
        ComputeBuffer _dataBuffer;
        ComputeBuffer _partialSumBuffer;
        ComputeBuffer _partialSumBufferRead;

        void Start()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Debug.LogError("Compute Shader is not Support!!");
                return;
            }
            if (_computeShader == null)
            {
                Debug.LogError("Compute Shader has not been assigned!!");
                return;
            }

            InitializeForGPUMode();
            InitializeForCPUMode();
        }

        void Update()
        {
            if (!SystemInfo.supportsComputeShaders)
            {
                Debug.LogError("Compute Shader is not Support!!");
                return;
            }
            if (_computeShader == null)
            {
                Debug.LogError("Compute Shader has not been assigned!!");
                return;
            }

            System.Diagnostics.Stopwatch _stopWatch = new System.Diagnostics.Stopwatch();
            _stopWatch.Start();

            if (_mode == ExecMode.GPU)
            {
                SetDataGPU();
                float gpuSum = ReduceGPU();
                Debug.Log("Sum of GPU = " + gpuSum);
            }
            else
            {
                SetDataCPU();
                float cpuSum = ReduceCPU();
                Debug.Log("Sum of CPU = " + cpuSum);
            }

            _stopWatch.Stop();
            Debug.Log("Elapsed time: " + _stopWatch.ElapsedMilliseconds + "ms");
        }

        void InitializeForGPUMode()
        {
            uint threadsPerGroupsX, threadsPerGroupsY, threadsPerGroupsZ;

            int kernelID = _computeShader.FindKernel(kernelNames[0]);
            _computeShader.GetKernelThreadGroupSizes(kernelID, 
                out threadsPerGroupsX, out threadsPerGroupsY, out threadsPerGroupsZ);

            _numOfGpuThreads = threadsPerGroupsX;
            _numOfGroups = Mathf.CeilToInt((float)_size / _numOfGpuThreads);

            _dataBuffer = new ComputeBuffer(_size, sizeof(float));
            _partialSumBuffer = new ComputeBuffer(_numOfGroups, sizeof(float));
            _partialSumBufferRead = new ComputeBuffer(_numOfGroups, sizeof(float));

            _partialSums = new float[_numOfGroups];
        }

        void SetDataGPU()
        {
            int kernelID = _computeShader.FindKernel(kernelNames[0]);
            _computeShader.SetBuffer(kernelID, "_DataArray", _dataBuffer);

            // The total number of execution threads is numOfGroups*numOfGpuThreads
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
        }

        float ReduceGPU()
        {
            int kernelID = _computeShader.FindKernel(kernelNames[1]);
            _computeShader.SetBuffer(kernelID, "_DataArray", _dataBuffer);
            _computeShader.SetBuffer(kernelID, "_PartialSumWrite", _partialSumBuffer);

            // The total number of execution threads is numOfGroups*numOfGpuThreads
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
            _partialSumBuffer.GetData(_partialSums);

            float sum = 0.0f;
            for (int k = 0; k < _numOfGroups; k++)
            {
                sum += _partialSums[k];
            }
    
            return sum;
        }

        float ReduceGPUV2()
        {
            // Debug.Log("Num of groups: " + _numOfGroups);

            int kernelID = _computeShader.FindKernel(kernelNames[1]);
            int nk = Mathf.CeilToInt((float)_size / _numOfGpuThreads);

            // Debug.Log("nk: " + nk);
            _computeShader.SetBuffer(kernelID, "_DataArray", _dataBuffer);
            _computeShader.SetBuffer(kernelID, "_PartialSumWrite", _partialSumBuffer);
            _computeShader.Dispatch(kernelID, nk, 1, 1);

            while(nk > 1)
            {
                kernelID = _computeShader.FindKernel("CsCopyBuffer");
                _computeShader.SetInt("_PartialSumSize", nk);
                _computeShader.SetBuffer(kernelID, "_PartialSumRead", _partialSumBuffer);
                _computeShader.SetBuffer(kernelID, "_PartialSumWrite", _partialSumBufferRead);
                _computeShader.Dispatch(kernelID, nk, 1, 1);

                nk = Mathf.CeilToInt((float)nk / _numOfGpuThreads);
                // Debug.Log("nk: " + nk);

                kernelID = _computeShader.FindKernel(kernelNames[1]);
                _computeShader.SetBuffer(kernelID, "_DataArray", _partialSumBufferRead);
                _computeShader.SetBuffer(kernelID, "_PartialSumWrite", _partialSumBuffer);
                _computeShader.Dispatch(kernelID, nk, 1, 1);
           }

            _partialSumBuffer.GetData(_partialSums);

            float sum = _partialSums[0];
    
            return sum;
        }

        #region CPU implementation

        void InitializeForCPUMode()
        {
            _dataArray = new float[_size];
        }

        void SetDataCPU()
        {
            for (int k = 0; k < _dataArray.Length; k++)
            {
                // Data length
                _dataArray[k] = 1.0f;

                // Leibniz formula for π
                // float sign = (k % 2 == 0) ? 1.0f : -1.0f;
                // _dataArray[k] = sign/(2.0f*k + 1.0f);
            }
        }

        float ReduceCPU()
        {
            float sum = 0.0f;
            for (int k = 0; k < _dataArray.Length; k++)
            {
                sum += _dataArray[k];
            }
            return sum;
        }

        #endregion
    }
}
