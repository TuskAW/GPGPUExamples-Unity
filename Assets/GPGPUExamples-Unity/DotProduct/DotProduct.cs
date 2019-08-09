using UnityEngine;

namespace GPGPUExamples
{
    public class DotProduct : MonoBehaviour
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
            "CsDotProduct",
        };

        uint _numOfGpuThreads; // number of threads per group
        int _numOfGroups;

        float[] _vectorA;
        float[] _vectorB;
        float[] _partialDot;
        ComputeBuffer _vectorBufferA;
        ComputeBuffer _vectorBufferB;
        ComputeBuffer _partialDotBuffer;

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
                float gpuSum = DotProductGPU();
                Debug.Log("Sum of GPU = " + gpuSum);
            }
            else
            {
                SetDataCPU();
                float cpuSum = DotProductCPU();
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

            _vectorBufferA = new ComputeBuffer(_size, sizeof(float));
            _vectorBufferB = new ComputeBuffer(_size, sizeof(float));
            _partialDotBuffer = new ComputeBuffer(_numOfGroups, sizeof(float));

            _partialDot = new float[_numOfGroups];
        }

        void SetDataGPU()
        {
            int kernelID = _computeShader.FindKernel(kernelNames[0]);
            _computeShader.SetBuffer(kernelID, "_VectorA", _vectorBufferA);
            _computeShader.SetBuffer(kernelID, "_VectorB", _vectorBufferB);

            // The total number of execution threads is numOfGroups*numOfGpuThreads
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
        }

        float DotProductGPU()
        {
            int kernelID = _computeShader.FindKernel(kernelNames[1]);
            _computeShader.SetBuffer(kernelID, "_VectorA", _vectorBufferA);
            _computeShader.SetBuffer(kernelID, "_VectorB", _vectorBufferB);
            _computeShader.SetBuffer(kernelID, "_PartialDot", _partialDotBuffer);

            // The total number of execution threads is numOfGroups*numOfGpuThreads
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
            _partialDotBuffer.GetData(_partialDot);

            float dotProduct = 0.0f;
            for (int k = 0; k < _numOfGroups; k++)
            {
                dotProduct += _partialDot[k];
            }
    
            return dotProduct;
        }

        #region CPU implementation

        void InitializeForCPUMode()
        {
            _vectorA = new float[_size];
            _vectorB = new float[_size];
        }

        void SetDataCPU()
        {
            for (int k = 0; k < _vectorA.Length; k++)
            {
                // Data length
                // _vectorA[k] = 1.0f;
                // _vectorB[k] = 2.0f;

                // Basel problem
                // The sum of the series is approximately equal to 1.644934 (pi^2/6)
                _vectorA[k] = 1.0f/(k+1);
                _vectorB[k] = 1.0f/(k+1);
            }
        }

        float DotProductCPU()
        {
            float sum = 0.0f;
            for (int k = 0; k < _vectorA.Length; k++)
            {
                sum += _vectorA[k]*_vectorB[k];
            }
            return sum;
        }

        #endregion
    }
}
