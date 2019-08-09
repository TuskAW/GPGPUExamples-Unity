using UnityEngine;

//========================================
// Solve the poisson equation L*v=b
// using Conjugate Gradient method
//========================================
//----- Conjugate Gradient algorithm -----
//
// Solve the linear equation: L*v=b
// Coefficient matrix (Laplacian): L
// Solution vector: v
// Source term vector: b
//
// Initialization
// set initial guess v0
// r0 = b - L*v0
// p0 = r0
//
// Iteration ( until |rk|/|b|<eps )
// alpha = (rk,rk)/(pk,L*pk)
// vk1 = vk + alpha*pk
// rk1 = rk - alpha*L*pk
// beta = (rk1,rk1)/(rk,rk)
// pk1 = rk1 + beta*pk
//
//----------------------------------------
namespace GPGPUExamples
{
    public class Poisson1D_GPU : MonoBehaviour
    {
        [SerializeField] ComputeShader _computeShader;
        [SerializeField] int _size = 2147483;
        [SerializeField] int _maxIter = 50;

        bool _initialized = false;

        uint _numOfGpuThreads; // number of threads per group
        int _numOfGroups;

        float[] _partialDot;
        ComputeBuffer _partialDotBuffer;

        ComputeBuffer _b; // source term
        ComputeBuffer _vk; // the solution of the poisson equation
        ComputeBuffer _rk;
        ComputeBuffer _pk;
        ComputeBuffer _Lp;
        ComputeBuffer _Lv;

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

            SolvePoissonGPU(_size);

            _stopWatch.Stop();
            Debug.Log("Elapsed time: " + _stopWatch.ElapsedMilliseconds + "ms");
        }

        void InitializeForGPUMode()
        {
            uint threadsPerGroupsX, threadsPerGroupsY, threadsPerGroupsZ;

            int kernelID = _computeShader.FindKernel("CsSetInitialData");
            _computeShader.GetKernelThreadGroupSizes(kernelID, 
                out threadsPerGroupsX, out threadsPerGroupsY, out threadsPerGroupsZ);

            _numOfGpuThreads = threadsPerGroupsX;
            _numOfGroups = Mathf.CeilToInt((float)_size / _numOfGpuThreads);

            _partialDot = new float[_numOfGroups];
            _partialDotBuffer = new ComputeBuffer(_numOfGroups, sizeof(float));

            _b = new ComputeBuffer(_size, sizeof(float));
            _vk = new ComputeBuffer(_size, sizeof(float));
            _rk = new ComputeBuffer(_size, sizeof(float));
            _pk = new ComputeBuffer(_size, sizeof(float));
            _Lp = new ComputeBuffer(_size, sizeof(float));
            _Lv = new ComputeBuffer(_size, sizeof(float));

            _initialized = true;
        }

        void SolvePoissonGPU(int n)
        {
            if (!_initialized)
            {
                InitializeForGPUMode();
            }

            CsSetInitialData(_vk, _b, n);
            CsLpMV1(_Lv, _vk, n);
            CsSzaxpy(_rk, -1.0f, _Lv, _b, n);
            CsCopyVector(_pk, _rk, n);

            // Iteration
            for (int iter = 0; iter < _maxIter; iter++)
            {
                float rkrk = CsDotProduct(_rk, _rk ,n);

                // Convergence check
                if (rkrk < Mathf.Epsilon)
                {
                    Debug.Log("Converged");
                    Debug.Log("Iter: " + iter);
                    Debug.Log("rkrk: " + rkrk);
                    break;
                }

                CsLpMV1(_Lp, _pk, n);
                float pLp = CsDotProduct(_pk, _Lp, n);
                float alpha = rkrk/pLp;

                CsSzaxpy(_rk, -alpha, _Lp, _rk, n);

                float beta = CsDotProduct(_rk, _rk, n)/rkrk;
                CsSzaxpy(_pk, beta, _pk, _rk, n);
            }
        }

        void CsSetInitialData(ComputeBuffer v0, ComputeBuffer b, int n)
        {
            int kernelID = _computeShader.FindKernel("CsSetInitialData");
            _computeShader.SetInt("_N", n);
            _computeShader.SetBuffer(kernelID, "_vk", v0); // Initial guess
            _computeShader.SetBuffer(kernelID, "_b", b); // Source term
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
        }

        void CsCopyVector(ComputeBuffer dst, ComputeBuffer src, int n)
        {
            int kernelID = _computeShader.FindKernel("CsCopyVector");
            _computeShader.SetInt("_N", n);
            _computeShader.SetBuffer(kernelID, "_y", src);
            _computeShader.SetBuffer(kernelID, "_x", dst);
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
        }

        void CsLpMV1(ComputeBuffer Lv, ComputeBuffer vk, int n)
        {
            int kernelID = _computeShader.FindKernel("CsLpMV1");
            _computeShader.SetInt("_N", n);
            _computeShader.SetBuffer(kernelID, "_vk", vk);
            _computeShader.SetBuffer(kernelID, "_Lv", Lv);
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
        }

        void CsSzaxpy(ComputeBuffer z, float alpha, ComputeBuffer x, ComputeBuffer y, int n)
        {
            int kernelID = _computeShader.FindKernel("CsSzaxpy");
            _computeShader.SetInt("_N", n);
            _computeShader.SetFloat("_alpha", alpha);
            _computeShader.SetBuffer(kernelID, "_x", x);
            _computeShader.SetBuffer(kernelID, "_y", y);
            _computeShader.SetBuffer(kernelID, "_z", z);
            _computeShader.Dispatch(kernelID, _numOfGroups, 1, 1);
        }

        float CsDotProduct(ComputeBuffer vectorA, ComputeBuffer vectorB, int n)
        {
            int kernelID = _computeShader.FindKernel("CsDotProduct");
            _computeShader.SetInt("_N", n);
            _computeShader.SetBuffer(kernelID, "_VectorA", vectorA);
            _computeShader.SetBuffer(kernelID, "_VectorB", vectorB);
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
    }
}
