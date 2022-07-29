/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <utility>
#include <vector>
/*
This testfile verifies the following scenarios of all hipMemcpy API
1. Negative Scenarios
2. Half Memory copy scenarios
3. Null check scenario
*/

static constexpr size_t NUM_ELM{1024 * 1024};

static constexpr size_t W{1024};
static constexpr size_t H{1024};
static constexpr size_t D{1};


// create a hipMemcpy3DParams struct for the 3d version of memcpy to verify the memset operation
hipMemcpy3DParms createParams(hipMemcpyKind kind, float* src, float* dst, size_t srcPitch,
                              size_t dstPitch, size_t dataW, size_t dataH, size_t dataD) {
  hipMemcpy3DParms p = {};
  p.kind = kind;

  p.srcPtr.ptr = src;
  p.srcPtr.pitch = srcPitch;
  p.srcPtr.xsize = dataW;
  p.srcPtr.ysize = dataH;

  p.dstPtr.ptr = dst;
  p.dstPtr.pitch = dstPitch;
  p.dstPtr.xsize = dataW;
  p.dstPtr.ysize = dataH;

  hipExtent extent = make_hipExtent(dataW * sizeof(float), dataH, dataD);
  p.extent = extent;

  return p;
}


/*This testcase verifies the negative scenarios of hipMemcpy APIs
 */
TEST_CASE("Unit_hipMemcpy_Negative") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  float* D_dPitched{nullptr};  // pitched ptr
  float* E_d3D{nullptr};       // 3D ptr

  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  SECTION("Pass nullptr to destination pointer for all Memcpy APIs") {
    HIP_CHECK_ERROR(hipMemcpy(nullptr, A_d, NUM_ELM * sizeof(float), hipMemcpyDefault),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyAsync(nullptr, A_h, NUM_ELM * sizeof(float), hipMemcpyDefault, stream),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyHtoD(hipDeviceptr_t(nullptr), A_h, NUM_ELM * sizeof(float)),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyHtoDAsync(hipDeviceptr_t(nullptr), A_h, NUM_ELM * sizeof(float), stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyDtoH(nullptr, hipDeviceptr_t(A_d), NUM_ELM * sizeof(float)),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyDtoHAsync(nullptr, hipDeviceptr_t(A_d), NUM_ELM * sizeof(float), stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyDtoD(hipDeviceptr_t(nullptr), hipDeviceptr_t(A_d), NUM_ELM * sizeof(float)),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyDtoDAsync(hipDeviceptr_t(nullptr), hipDeviceptr_t(A_d),
                                       NUM_ELM * sizeof(float), stream),
                    hipErrorInvalidValue);
    // 2D
    size_t dataPitch{};
    HIP_CHECK(
        hipMallocPitch(reinterpret_cast<void**>(&D_dPitched), &dataPitch, W * sizeof(float), H));

    HIP_CHECK_ERROR(
        hipMemcpy2DAsync(nullptr, W, D_dPitched, dataPitch, W, H, hipMemcpyDefault, stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpy2D(nullptr, W, D_dPitched, dataPitch, W, H, hipMemcpyDefault),
                    hipErrorInvalidValue);
    // 3D
    hipExtent extent = make_hipExtent(W * sizeof(float), H, D);
    hipPitchedPtr pitchedAPtr;
    HIP_CHECK(hipMalloc3D(&pitchedAPtr, extent));
    dataPitch = pitchedAPtr.pitch;
    E_d3D = reinterpret_cast<float*>(pitchedAPtr.ptr);


    hipMemcpy3DParms params = createParams(hipMemcpyDefault, E_d3D, nullptr, dataPitch,
                                           static_cast<size_t>(W * sizeof(float)), W, H, D);
    HIP_CHECK_ERROR(hipMemcpy3DAsync(&params, stream), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpy3D(&params), hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to source pointer") {
    HIP_CHECK_ERROR(hipMemcpy(A_h, nullptr, NUM_ELM * sizeof(float), hipMemcpyDefault),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyAsync(A_d, nullptr, NUM_ELM * sizeof(float), hipMemcpyDefault, stream),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyHtoD(hipDeviceptr_t(A_d), nullptr, NUM_ELM * sizeof(float)),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), nullptr, NUM_ELM * sizeof(float), stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyDtoH(A_h, hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float)),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyDtoHAsync(A_h, hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float), stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyDtoD(hipDeviceptr_t(A_d), hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float)),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d), hipDeviceptr_t(nullptr),
                                       NUM_ELM * sizeof(float), stream),
                    hipErrorInvalidValue);
    // 2D
    size_t dataPitch{};
    HIP_CHECK(
        hipMallocPitch(reinterpret_cast<void**>(&D_dPitched), &dataPitch, W * sizeof(float), H));

    HIP_CHECK_ERROR(
        hipMemcpy2DAsync(D_dPitched, dataPitch, nullptr, W, W, H, hipMemcpyDefault, stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpy2D(D_dPitched, dataPitch, nullptr, W, W, H, hipMemcpyDefault),
                    hipErrorInvalidValue);
    // 3D
    hipExtent extent = make_hipExtent(W * sizeof(float), H, D);
    hipPitchedPtr pitchedAPtr;
    HIP_CHECK(hipMalloc3D(&pitchedAPtr, extent));
    dataPitch = pitchedAPtr.pitch;
    E_d3D = reinterpret_cast<float*>(pitchedAPtr.ptr);


    hipMemcpy3DParms params = createParams(hipMemcpyDefault, nullptr, E_d3D, W, dataPitch, W, H, D);
    HIP_CHECK_ERROR(hipMemcpy3DAsync(&params, stream), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpy3D(&params), hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to both source and dest pointer") {
    HIP_CHECK_ERROR(hipMemcpy(nullptr, nullptr, NUM_ELM * sizeof(float), hipMemcpyDefault),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyAsync(nullptr, nullptr, NUM_ELM * sizeof(float), hipMemcpyDefault, stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyHtoD(hipDeviceptr_t(nullptr), nullptr, NUM_ELM * sizeof(float)),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyHtoDAsync(hipDeviceptr_t(nullptr), nullptr, NUM_ELM * sizeof(float), stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyDtoH(nullptr, hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float)),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyDtoHAsync(nullptr, hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float), stream),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipMemcpyDtoD(hipDeviceptr_t(nullptr), hipDeviceptr_t(nullptr), NUM_ELM * sizeof(float)),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpyDtoDAsync(hipDeviceptr_t(nullptr), hipDeviceptr_t(nullptr),
                                       NUM_ELM * sizeof(float), stream),
                    hipErrorInvalidValue);
    // 2D
    HIP_CHECK_ERROR(hipMemcpy2DAsync(nullptr, W, nullptr, W, W, H, hipMemcpyDefault, stream),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpy2D(nullptr, W, nullptr, W, W, H, hipMemcpyDefault),
                    hipErrorInvalidValue);
    // 3D
    hipMemcpy3DParms params = createParams(hipMemcpyDefault, nullptr, nullptr, W, W, W, H, D);
    HIP_CHECK_ERROR(hipMemcpy3DAsync(&params, stream), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemcpy3D(&params), hipErrorInvalidValue);
  }

  SECTION("Passing same pointers") {
    HIP_CHECK(hipMemcpy(A_d, A_d, (NUM_ELM / 2) * sizeof(float), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(A_h, A_h, (NUM_ELM / 2) * sizeof(float), hipMemcpyDefault));
    HIP_CHECK(hipMemcpyAsync(A_d, A_d, (NUM_ELM / 2) * sizeof(float), hipMemcpyDefault, stream));
    HIP_CHECK(hipMemcpyAsync(A_h, A_h, (NUM_ELM / 2) * sizeof(float), hipMemcpyDefault, stream));
    HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(A_d), hipDeviceptr_t(A_d), NUM_ELM * sizeof(float)));
    HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(A_d), hipDeviceptr_t(A_d), NUM_ELM * sizeof(float),
                                 stream));
    // 2D
    size_t dataPitch{};
    HIP_CHECK(
        hipMallocPitch(reinterpret_cast<void**>(&D_dPitched), &dataPitch, W * sizeof(float), H));

    HIP_CHECK(hipMemcpy2DAsync(D_dPitched, dataPitch, D_dPitched, dataPitch, W, H, hipMemcpyDefault,
                               stream));
    HIP_CHECK(hipMemcpy2D(D_dPitched, dataPitch, D_dPitched, dataPitch, W, H, hipMemcpyDefault));
    // 3D
    hipExtent extent = make_hipExtent(W * sizeof(float), H, D);
    hipPitchedPtr pitchedAPtr;
    HIP_CHECK(hipMalloc3D(&pitchedAPtr, extent));
    dataPitch = pitchedAPtr.pitch;
    E_d3D = reinterpret_cast<float*>(pitchedAPtr.ptr);

    hipMemcpy3DParms params =
        createParams(hipMemcpyDefault, E_d3D, E_d3D, dataPitch, dataPitch, W, H, D);
    HIP_CHECK(hipMemcpy3DAsync(&params, stream));
    HIP_CHECK(hipMemcpy3D(&params));
  }

  HipTest::freeArrays<float>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipFree(D_dPitched));
  HIP_CHECK(hipFree(E_d3D));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*
This testcase verifies the Nullcheck for all the 8 Memcpy APIs
*/
TEST_CASE("Unit_hipMemcpy_NullCheck") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  float* D_dPitched{nullptr};  // pitched ptr
  float* E_d3D{nullptr};       // 3D ptr
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM);
  hipStream_t stream;
  hipStreamCreate(&stream);
  HIP_CHECK(hipMemcpy(A_d, C_h, NUM_ELM * sizeof(float), hipMemcpyHostToDevice));

  SECTION("hipMemcpyHtoD API null size check") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h, 0));
    HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }

  SECTION("hipMemcpyHtoDAsync API null size check") {
    HIP_CHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), A_h, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpy API null size check") {
    HIP_CHECK(hipMemcpy(A_d, B_h, 0, hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyAsync API null size check") {
    HIP_CHECK(hipMemcpyAsync(A_d, B_h, 0, hipMemcpyDefault, stream));
    HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoH API null size check") {
    HIP_CHECK(hipMemcpyDtoH(C_h, hipDeviceptr_t(A_d), 0));
    HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoHAsync API null size check") {
    HIP_CHECK(hipMemcpyDtoHAsync(C_h, hipDeviceptr_t(A_d), 0, stream));
    HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(C_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoD API null size check") {
    HIP_CHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyDtoD(hipDeviceptr_t(C_d), hipDeviceptr_t(A_d), 0));
    HIP_CHECK(hipMemcpy(B_h, C_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM);
  }
  SECTION("hipMemcpyDtoDAsync API null size check") {
    HIP_CHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(C_d), hipDeviceptr_t(A_d), 0, stream));
    HIP_CHECK(hipMemcpy(B_h, C_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM);
  }

  size_t dataPitch{};
  SECTION("hipMemcpy2D API null size check") {
    // copy into A_d
    HIP_CHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMallocPitch(reinterpret_cast<void**>(&D_dPitched), &dataPitch, W * sizeof(float), H));
    SECTION("hipMemcpy2D ") {
      // do a false (0 size) copy into A_d
      HIP_CHECK(
          hipMemcpy2D(C_d, NUM_ELM * sizeof(float), D_dPitched, dataPitch, 0, 0, hipMemcpyDefault));
      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, C_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM);
    }
    SECTION("hipMemcpy2DAsync") {
      HIP_CHECK(hipMemcpy2DAsync(C_d, NUM_ELM * sizeof(float), D_dPitched, dataPitch, 0, 0,
                                 hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, C_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM);
    }
  }
  SECTION("hipMemcpy3D API null size check") {
    HIP_CHECK(hipMemcpy(C_d, A_h, NUM_ELM * sizeof(float), hipMemcpyHostToDevice));
    hipExtent extent = make_hipExtent(W * sizeof(float), H, D);
    hipPitchedPtr pitchedAPtr;
    HIP_CHECK(hipMalloc3D(&pitchedAPtr, extent));
    dataPitch = pitchedAPtr.pitch;
    E_d3D = reinterpret_cast<float*>(pitchedAPtr.ptr);

    hipMemcpy3DParms params =
        createParams(hipMemcpyDefault, E_d3D, C_d, dataPitch, dataPitch, 0, 0, 0);
    SECTION("hipMemcpy3D ") {
      HIP_CHECK(hipMemcpy3D(&params));
      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, C_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM);
    }
    SECTION("hipMemcpy3DAsync ") {
      HIP_CHECK(hipMemcpy3DAsync(&params, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, C_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM);
    }
  }

  HipTest::freeArrays<float>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipFree(D_dPitched));
  HIP_CHECK(hipFree(E_d3D));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*
This testcase verifies all the hipMemcpy APIs by
copying half the memory.
*/
TEST_CASE("Unit_hipMemcpy_HalfMemCopy") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  float* D_dPitched{nullptr};  // pitched ptr
  float* E_d3D{nullptr};       // 3D ptr
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM);
  hipStream_t stream;
  hipStreamCreate(&stream);

  SECTION("hipMemcpyHtoD half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h, (NUM_ELM * sizeof(float)) / 2));
    HIP_CHECK(hipMemcpy(B_h, A_d, (NUM_ELM * sizeof(float)) / 2, hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }

  SECTION("hipMemcpyHtoDAsync half memory copy") {
    HIP_CHECK(hipMemcpyHtoDAsync(hipDeviceptr_t(A_d), A_h, (NUM_ELM * sizeof(float)) / 2, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(B_h, A_d, (NUM_ELM * sizeof(float)) / 2, hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }

  SECTION("hipMemcpyDtoH half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h, (NUM_ELM * sizeof(float))));
    HIP_CHECK(hipMemcpyDtoH(B_h, hipDeviceptr_t(A_d), (NUM_ELM * sizeof(float)) / 2));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }

  SECTION("hipMemcpyDtoHAsync half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h, (NUM_ELM * sizeof(float))));
    HIP_CHECK(hipMemcpyDtoHAsync(B_h, hipDeviceptr_t(A_d), (NUM_ELM * sizeof(float)) / 2, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }

  SECTION("hipMemcpyDtoD half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h, (NUM_ELM * sizeof(float)) / 2));
    HIP_CHECK(
        hipMemcpyDtoD(hipDeviceptr_t(B_d), hipDeviceptr_t(A_d), (NUM_ELM * sizeof(float)) / 2));
    HIP_CHECK(hipMemcpy(B_h, B_d, (NUM_ELM * sizeof(float)) / 2, hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }

  SECTION("hipMemcpyDtoDAsync half memory copy") {
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(A_d), A_h, (NUM_ELM * sizeof(float)) / 2));
    HIP_CHECK(hipMemcpyDtoDAsync(hipDeviceptr_t(B_d), hipDeviceptr_t(A_d),
                                 (NUM_ELM * sizeof(float)) / 2, stream));
    HIP_CHECK(hipMemcpy(B_h, B_d, (NUM_ELM * sizeof(float)) / 2, hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }

  SECTION("hipMemcpy half memory copy") {
    HIP_CHECK(hipMemcpy(A_d, A_h, (NUM_ELM * sizeof(float)), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(B_h, A_d, (NUM_ELM / 2) * sizeof(float), hipMemcpyDeviceToHost));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }

  SECTION("hipMemcpyAsync half memory copy") {
    HIP_CHECK(hipMemcpy(A_d, A_h, (NUM_ELM * sizeof(float)), hipMemcpyDefault));
    HIP_CHECK(
        hipMemcpyAsync(B_h, A_d, (NUM_ELM / 2) * sizeof(float), hipMemcpyDeviceToHost, stream));
    HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
  }


  // TODO: 2D and 3D Memcpy for half size return invalid value (Could not find cause)
  size_t dataPitch{};
  SECTION("hipMemcpy2D API memory copy") {
    // copy into A_d
    HIP_CHECK(hipMemcpy(A_d, A_h, NUM_ELM * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMallocPitch(reinterpret_cast<void**>(&D_dPitched), &dataPitch, W * sizeof(float), H));
    SECTION("hipMemcpy2D ") {
      HIP_CHECK(hipMemcpy2D(A_d, NUM_ELM * sizeof(float), D_dPitched, dataPitch, W * sizeof(float),
                            H / 2, hipMemcpyDeviceToDevice));
      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM / 2 * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
    }
    SECTION("hipMemcpy2DAsync") {
      HIP_CHECK(hipMemcpy2DAsync(A_d, NUM_ELM * sizeof(float), D_dPitched, dataPitch,
                                 W * sizeof(float), H / 2, hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM / 2 * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM / 2);
    }
  }
  SECTION("hipMemcpy3D API memory copy") {
    HIP_CHECK(hipMemcpy(A_d, A_h, NUM_ELM * sizeof(float), hipMemcpyHostToDevice));
    hipExtent extent = make_hipExtent(W * sizeof(float), H, D);
    hipPitchedPtr pitchedAPtr;
    HIP_CHECK(hipMalloc3D(&pitchedAPtr, extent));
    dataPitch = pitchedAPtr.pitch;
    E_d3D = reinterpret_cast<float*>(pitchedAPtr.ptr);

    hipMemcpy3DParms params =
        createParams(hipMemcpyDefault, E_d3D, A_d, dataPitch, NUM_ELM * sizeof(float), W, H / 2, D);
    SECTION("hipMemcpy3D ") {
      HIP_CHECK(hipMemcpy3D(&params));
      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM);
    }
    SECTION("hipMemcpy3DAsync ") {
      HIP_CHECK(hipMemcpy3DAsync(&params, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      // copy data back to host to verify
      HIP_CHECK(hipMemcpy(B_h, A_d, NUM_ELM * sizeof(float), hipMemcpyDeviceToHost));
      HipTest::checkTest(A_h, B_h, NUM_ELM);
    }
  }
  HipTest::freeArrays<float>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipFree(D_dPitched));
  HIP_CHECK(hipFree(E_d3D));
  HIP_CHECK(hipStreamDestroy(stream));
}
// TODO: Extend testing for following scenairos

/*
This testcase verifies all the hipMemcpy APIs by
copying more than allocated memory.
*/
TEST_CASE("Unit_hipMemcpy_CopyTooBig") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM);
}

/*
This testcase verifies all the hipMemcpy APIs by
copying memory at an incorrect offset.
*/
TEST_CASE("Unit_hipMemcpy_BadOffset") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM);
}
/*
This testcase verifies all the hipMemcpy APIs by
copying with incorrect type.
*/
TEST_CASE("Unit_hipMemcpy_InccorectType") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM);
}
/*
This testcase verifies all the hipMemcpyAsync APIs by
copying using an invalid stream.
*/
TEST_CASE("Unit_hipMemcpy_InvalidStream") {
  // Initialization of variables
  float *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  float *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HIP_CHECK(hipSetDevice(0));
  HipTest::initArrays<float>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, NUM_ELM);
}
