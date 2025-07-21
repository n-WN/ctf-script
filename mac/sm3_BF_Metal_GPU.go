package main

// 感谢 rec https://github.com/Rechn0 的队友的思路

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreGraphics

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Metal 设备和资源
id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;
id<MTLComputePipelineState> computePipelineState;
id<MTLBuffer> candidateBuffer;
id<MTLBuffer> resultBuffer;
id<MTLBuffer> targetBuffer;
id<MTLBuffer> foundBuffer;

// SM3 Metal shader 源码
const char* sm3MetalSource = R"(
#include <metal_stdlib>
using namespace metal;

// SM3 常量
constant uint32_t SM3_IV[8] = {
    0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
    0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e
};

// 循环左移
inline uint32_t rotateLeft(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// SM3 函数
inline uint32_t ff0(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
inline uint32_t ff1(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (x & z) | (y & z); }
inline uint32_t gg0(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
inline uint32_t gg1(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
inline uint32_t p0(uint32_t x) { return x ^ rotateLeft(x, 9) ^ rotateLeft(x, 17); }
inline uint32_t p1(uint32_t x) { return x ^ rotateLeft(x, 15) ^ rotateLeft(x, 23); }

// 字符映射
inline uchar indexToChar(uint64_t index) {
    const uchar chars[4] = {'a', 'b', 'c', 'd'};
    return chars[index & 3];
}

// SM3 核心计算 - 使用线程本地内存
void sm3_hash_local(thread const uchar* input, thread uchar* output) {
    uint32_t digest[8];
    for (int i = 0; i < 8; i++) {
        digest[i] = SM3_IV[i];
    }

    // 准备消息块
    uint32_t W[68];
    uint32_t W1[64];

    // 填充消息
    uchar padded[64];
    for (int i = 0; i < 32; i++) {
        padded[i] = input[i];
    }
    padded[32] = 0x80;
    for (int i = 33; i < 62; i++) {
        padded[i] = 0;
    }
    padded[62] = 0x01;
    padded[63] = 0x00;

    // 消息扩展
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)padded[i*4] << 24) |
               ((uint32_t)padded[i*4+1] << 16) |
               ((uint32_t)padded[i*4+2] << 8) |
               ((uint32_t)padded[i*4+3]);
    }

    for (int i = 16; i < 68; i++) {
        W[i] = p1(W[i-16] ^ W[i-9] ^ rotateLeft(W[i-3], 15)) ^
               rotateLeft(W[i-13], 7) ^ W[i-6];
    }

    for (int i = 0; i < 64; i++) {
        W1[i] = W[i] ^ W[i+4];
    }

    // 压缩函数
    uint32_t A = digest[0], B = digest[1], C = digest[2], D = digest[3];
    uint32_t E = digest[4], F = digest[5], G = digest[6], H = digest[7];

    for (int i = 0; i < 16; i++) {
        uint32_t SS1 = rotateLeft(rotateLeft(A, 12) + E + rotateLeft(0x79cc4519, i), 7);
        uint32_t SS2 = SS1 ^ rotateLeft(A, 12);
        uint32_t TT1 = ff0(A, B, C) + D + SS2 + W1[i];
        uint32_t TT2 = gg0(E, F, G) + H + SS1 + W[i];
        D = C;
        C = rotateLeft(B, 9);
        B = A;
        A = TT1;
        H = G;
        G = rotateLeft(F, 19);
        F = E;
        E = p0(TT2);
    }

    for (int i = 16; i < 64; i++) {
        uint32_t SS1 = rotateLeft(rotateLeft(A, 12) + E + rotateLeft(0x7a879d8a, i), 7);
        uint32_t SS2 = SS1 ^ rotateLeft(A, 12);
        uint32_t TT1 = ff1(A, B, C) + D + SS2 + W1[i];
        uint32_t TT2 = gg1(E, F, G) + H + SS1 + W[i];
        D = C;
        C = rotateLeft(B, 9);
        B = A;
        A = TT1;
        H = G;
        G = rotateLeft(F, 19);
        F = E;
        E = p0(TT2);
    }

    // 最终哈希值
    digest[0] ^= A; digest[1] ^= B; digest[2] ^= C; digest[3] ^= D;
    digest[4] ^= E; digest[5] ^= F; digest[6] ^= G; digest[7] ^= H;

    // 输出大端序
    for (int i = 0; i < 8; i++) {
        output[i*4] = (digest[i] >> 24) & 0xff;
        output[i*4+1] = (digest[i] >> 16) & 0xff;
        output[i*4+2] = (digest[i] >> 8) & 0xff;
        output[i*4+3] = digest[i] & 0xff;
    }
}

// GPU 内核函数
kernel void sm3_search(
    device uchar* result [[buffer(0)]],      // 输出结果
    constant uchar* target [[buffer(1)]],    // 目标哈希
    device atomic_int* found [[buffer(2)]],  // 找到标志
    constant uint64_t* baseIndex [[buffer(3)]], // 基础索引
    uint3 gid [[thread_position_in_grid]]    // 线程ID
) {
    // 计算全局索引
    uint64_t globalId = gid.x + gid.y * 1024 + gid.z * 1024 * 1024;
    uint64_t candidateIndex = baseIndex[0] + globalId;

    // 检查是否已找到
    if (atomic_load_explicit(found, memory_order_relaxed) != 0) {
        return;
    }

    // 生成候选值 - 使用线程本地内存
    thread uchar candidate[32];

    // 固定前缀 "adcddbbadcacabad"
    candidate[0] = 'a'; candidate[1] = 'd'; candidate[2] = 'c'; candidate[3] = 'd';
    candidate[4] = 'd'; candidate[5] = 'b'; candidate[6] = 'b'; candidate[7] = 'a';
    candidate[8] = 'd'; candidate[9] = 'c'; candidate[10] = 'a'; candidate[11] = 'c';
    candidate[12] = 'a'; candidate[13] = 'b'; candidate[14] = 'a'; candidate[15] = 'd';

    // 生成后16字节
    uint64_t idx = candidateIndex;
    for (int i = 0; i < 16; i++) {
        candidate[16 + i] = indexToChar(idx);
        idx >>= 2;
    }

    // 计算哈希 - 使用线程本地内存
    thread uchar hash[32];
    sm3_hash_local(candidate, hash);

    // 比较结果
    bool match = true;
    for (int i = 0; i < 32; i++) {
        if (hash[i] != target[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        // 找到了！
        atomic_store_explicit(found, 1, memory_order_relaxed);

        // 保存结果到全局内存
        for (int i = 0; i < 32; i++) {
            result[i] = candidate[i];
        }
    }
}
)";

// 获取GPU信息
typedef struct {
    int coreCount;
    int maxThreadsPerThreadgroup;
    int maxThreadgroupsPerMeshGrid;
    int registryID;
    char name[256];
} GPUInfo;

// 使用system_profiler获取准确的GPU核心数
int getGPUCoresFromSystemProfiler() {
    FILE *fp;
    char buffer[128];
    int cores = 0;

    // 执行system_profiler命令
    fp = popen("system_profiler SPDisplaysDataType | awk '/Total Number of Cores:/{print $5}'", "r");
    if (fp == NULL) {
        printf("Failed to run system_profiler command\n");
        return 0;
    }

    // 读取输出
    if (fgets(buffer, sizeof(buffer), fp) != NULL) {
        cores = atoi(buffer);
    }

    pclose(fp);
    return cores;
}

GPUInfo getGPUInfo() {
    GPUInfo info = {0};

    if (device) {
        // GPU名称
        strncpy(info.name, [[device name] UTF8String], 255);

        // 使用system_profiler获取准确的核心数
        info.coreCount = getGPUCoresFromSystemProfiler();

        // 如果获取失败，使用保守的默认值
        if (info.coreCount == 0) {
            printf("Warning: Could not detect GPU cores, using default value\n");
            info.coreCount = 8; // 保守估计
        }

        // 获取注册表ID
        info.registryID = (int)[device registryID];
    }

    return info;
}

// 初始化 Metal
int initMetal(GPUInfo* gpuInfo) {
    @autoreleasepool {
        NSError *error = nil;

        // 获取所有GPU设备
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (devices.count > 0) {
            printf("Found %lu GPU devices:\n", devices.count);
            for (int i = 0; i < devices.count; i++) {
                id<MTLDevice> dev = devices[i];
                printf("  %d: %s\n", i, [[dev name] UTF8String]);
            }
            // 使用第一个设备（通常是最强大的）
            device = devices[0];
        } else {
            // 获取默认GPU设备
            device = MTLCreateSystemDefaultDevice();
        }

        if (!device) {
            printf("Metal is not supported on this device\n");
            return -1;
        }

        // 获取GPU详细信息
        *gpuInfo = getGPUInfo();

        printf("\n=== GPU Information ===\n");
        printf("GPU: %s\n", gpuInfo->name);
        printf("GPU Cores: %d\n", gpuInfo->coreCount);
        printf("Registry ID: %d\n", gpuInfo->registryID);

        // 输出GPU能力
        printf("\nGPU Capabilities:\n");
        printf("  Unified Memory: %s\n", [device hasUnifiedMemory] ? "YES" : "NO");
        printf("  Max Buffer Length: %.2f GB\n", (double)[device maxBufferLength] / (1024*1024*1024));
        printf("  Max Threads Per Threadgroup: %lu x %lu x %lu\n",
               [device maxThreadsPerThreadgroup].width,
               [device maxThreadsPerThreadgroup].height,
               [device maxThreadsPerThreadgroup].depth);

        if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
            printf("  Recommended Max Working Set: %.2f GB\n",
                   (double)[device recommendedMaxWorkingSetSize] / (1024*1024*1024));
        }

        // GPU Family支持
        printf("\nGPU Family Support:\n");
        if ([device supportsFamily:MTLGPUFamilyApple8]) {
            printf("  Apple GPU Family 8 (M2)\n");
        } else if ([device supportsFamily:MTLGPUFamilyApple7]) {
            printf("  Apple GPU Family 7 (M1)\n");
        }

        // 创建命令队列
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            printf("Failed to create command queue\n");
            return -1;
        }

        // 编译着色器
        NSString *source = [NSString stringWithUTF8String:sm3MetalSource];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            printf("Failed to compile shader: %s\n", [[error description] UTF8String]);
            return -1;
        }

        // 获取内核函数
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"sm3_search"];
        if (!kernelFunction) {
            printf("Failed to find kernel function\n");
            return -1;
        }

        // 创建计算管线状态
        computePipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!computePipelineState) {
            printf("Failed to create pipeline state: %s\n", [[error description] UTF8String]);
            return -1;
        }

        // 获取最大线程组大小
        gpuInfo->maxThreadsPerThreadgroup = (int)computePipelineState.maxTotalThreadsPerThreadgroup;
        printf("\nPipeline Info:\n");
        printf("  Max Threads Per Threadgroup: %d\n", gpuInfo->maxThreadsPerThreadgroup);
        printf("  Thread Execution Width: %lu\n", computePipelineState.threadExecutionWidth);

        // 创建缓冲区
        resultBuffer = [device newBufferWithLength:32 options:MTLResourceStorageModeShared];
        targetBuffer = [device newBufferWithLength:32 options:MTLResourceStorageModeShared];
        foundBuffer = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        candidateBuffer = [device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];

        if (!resultBuffer || !targetBuffer || !foundBuffer || !candidateBuffer) {
            printf("Failed to create buffers\n");
            return -1;
        }

        return 0;
    }
}

// 在GPU上搜索
int searchOnGPU(uint64_t startIndex, uint64_t count, const uint8_t* target, uint8_t* result, int maxThreadsPerThreadgroup) {
    @autoreleasepool {
        // 设置目标哈希
        memcpy([targetBuffer contents], target, 32);

        // 设置基础索引
        *(uint64_t*)[candidateBuffer contents] = startIndex;

        // 重置找到标志
        *(int*)[foundBuffer contents] = 0;

        // 创建命令缓冲区
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            printf("Failed to create command buffer\n");
            return -1;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            printf("Failed to create compute encoder\n");
            return -1;
        }

        [encoder setComputePipelineState:computePipelineState];
        [encoder setBuffer:resultBuffer offset:0 atIndex:0];
        [encoder setBuffer:targetBuffer offset:0 atIndex:1];
        [encoder setBuffer:foundBuffer offset:0 atIndex:2];
        [encoder setBuffer:candidateBuffer offset:0 atIndex:3];

        // 计算线程组大小 - 根据GPU能力动态调整
        NSUInteger threadsPerThreadgroup = MIN(maxThreadsPerThreadgroup, 256);
        if (threadsPerThreadgroup > computePipelineState.maxTotalThreadsPerThreadgroup) {
            threadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup;
        }

        NSUInteger threadgroupsPerGrid = (count + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

        // 限制总线程组数
        if (threadgroupsPerGrid > 65536) {
            threadgroupsPerGrid = 65536;
        }

        MTLSize threadsPerThreadgroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        MTLSize threadgroupsPerGridSize = MTLSizeMake(threadgroupsPerGrid, 1, 1);

        // 分发计算
        [encoder dispatchThreadgroups:threadgroupsPerGridSize
                threadsPerThreadgroup:threadsPerThreadgroupSize];

        [encoder endEncoding];

        // 提交并等待完成
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // 检查结果
        if (*(int*)[foundBuffer contents] != 0) {
            memcpy(result, [resultBuffer contents], 32);
            return 1;
        }

        return 0;
    }
}

// 清理资源
void cleanupMetal() {
    device = nil;
    commandQueue = nil;
    computePipelineState = nil;
    resultBuffer = nil;
    targetBuffer = nil;
    foundBuffer = nil;
    candidateBuffer = nil;
}
*/
import "C"

import (
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/schollz/progressbar/v3"
)

// GPU 配置（动态获取）
var (
	GPUCores                 int
	MaxThreadsPerThreadgroup int
	GPUBatchSize             int
)

var (
	tarHex   = "aab05fca300811223b3b957bfe33130770fb7a6b55b030a5809c559344f66f79"
	tarBytes []byte
)

var (
	globalProgress atomic.Int64
	foundFlag      atomic.Int32
	foundResult    [32]byte
	resultMutex    sync.Mutex
)

func init() {
	var err error
	tarBytes, err = hex.DecodeString(tarHex)
	if err != nil {
		log.Fatalf("无法解码目标哈希: %v", err)
	}

	// 初始化 Metal 并获取GPU信息
	fmt.Println("初始化 Metal GPU...")
	var gpuInfo C.GPUInfo
	if ret := C.initMetal(&gpuInfo); ret != 0 {
		log.Fatalf("Metal 初始化失败")
	}

	// 设置GPU参数
	GPUCores = int(gpuInfo.coreCount)
	MaxThreadsPerThreadgroup = int(gpuInfo.maxThreadsPerThreadgroup)

	// 计算最优批处理大小
	// 考虑GPU核心数和最大线程数
	GPUBatchSize = GPUCores * MaxThreadsPerThreadgroup * 16 // 16倍过度订阅
	if GPUBatchSize > (1 << 22) {                           // 最大4M
		GPUBatchSize = 1 << 22
	}

	fmt.Printf("\n=== GPU配置 ===\n")
	fmt.Printf("GPU核心数: %d\n", GPUCores)
	fmt.Printf("最大线程组大小: %d\n", MaxThreadsPerThreadgroup)
	fmt.Printf("批处理大小: %d (%.2fM)\n", GPUBatchSize, float64(GPUBatchSize)/(1024*1024))
	fmt.Println("\nMetal GPU 初始化成功！")
}

func main() {
	// 使用所有CPU核心协调GPU任务
	runtime.GOMAXPROCS(runtime.NumCPU())

	totalOperations := int64(256 * (0xffffff + 1))
	bar := progressbar.NewOptions64(totalOperations,
		progressbar.OptionSetDescription(fmt.Sprintf("GPU加速版 (%d核GPU)...", GPUCores)),
		progressbar.OptionShowBytes(false),
		progressbar.OptionSetWidth(30),
		progressbar.OptionShowCount(),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer: "=", SaucerHead: ">", SaucerPadding: " ",
			BarStart: "[", BarEnd: "]",
		}),
		progressbar.OptionThrottle(50*time.Millisecond),
	)

	// 创建任务队列
	jobs := make(chan uint64, 256)
	ctx, cancel := context.WithCancel(context.Background())

	wg := &sync.WaitGroup{}

	// 进度更新
	progressDone := make(chan struct{})
	go progressUpdater(bar, progressDone)

	// 启动GPU调度器
	numSchedulers := 4 // 使用4个调度器管理GPU任务
	for i := 0; i < numSchedulers; i++ {
		wg.Add(1)
		go gpuScheduler(i, wg, ctx, jobs)
	}

	timeStart := time.Now()

	// 分发任务
	fmt.Printf("\n正在使用 %d核GPU 进行并行计算...\n", GPUCores)
	fmt.Printf("每批次并行线程数: %d (%.2fM)\n", GPUBatchSize, float64(GPUBatchSize)/(1024*1024))
	fmt.Printf("最大线程组大小: %d\n\n", MaxThreadsPerThreadgroup)

	go func() {
		for j := uint64(0); j <= 0xff; j++ {
			select {
			case jobs <- j:
			case <-ctx.Done():
				return
			}
		}
		close(jobs)
	}()

	wg.Wait()
	cancel()
	close(progressDone)

	// 清理 Metal 资源
	C.cleanupMetal()

	timeEnd := time.Now()
	bar.Finish()

	duration := timeEnd.Sub(timeStart)
	totalHashes := globalProgress.Load()
	hashesPerSecond := float64(totalHashes) / duration.Seconds()

	fmt.Printf("\n=== GPU 性能统计 ===\n")
	fmt.Printf("GPU: %d核\n", GPUCores)
	fmt.Printf("总耗时: %v\n", duration)
	fmt.Printf("总哈希数: %d\n", totalHashes)
	fmt.Printf("哈希速率: %.2f MH/s\n", hashesPerSecond/1000000)
	fmt.Printf("每核心速率: %.2f MH/s\n", hashesPerSecond/1000000/float64(GPUCores))
	fmt.Printf("GPU吞吐量: %.2f GB/s\n", (hashesPerSecond*64)/(1024*1024*1024))

	if foundFlag.Load() != 0 {
		fmt.Printf("\n找到的结果: %s\n", string(foundResult[:]))
	}
}

func progressUpdater(bar *progressbar.ProgressBar, done <-chan struct{}) {
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	var lastProgress int64
	var lastTime time.Time = time.Now()
	var lastHashes int64

	for {
		select {
		case <-ticker.C:
			current := globalProgress.Load()
			if current > lastProgress {
				bar.Add64(current - lastProgress)

				// 计算实时速率
				now := time.Now()
				elapsed := now.Sub(lastTime).Seconds()
				if elapsed > 1.0 {
					rate := float64(current-lastHashes) / elapsed / 1000000
					bar.Describe(fmt.Sprintf("GPU计算中 (%.2f MH/s)...", rate))
					lastTime = now
					lastHashes = current
				}

				lastProgress = current
			}
		case <-done:
			current := globalProgress.Load()
			if current > lastProgress {
				bar.Add64(current - lastProgress)
			}
			return
		}
	}
}

func gpuScheduler(id int, wg *sync.WaitGroup, ctx context.Context, jobs <-chan uint64) {
	defer wg.Done()

	result := make([]byte, 32)

	for j := range jobs {
		if foundFlag.Load() != 0 {
			break
		}

		// 处理一个大任务块
		remaining := uint64(0xffffff + 1)
		offset := uint64(0)

		for remaining > 0 && foundFlag.Load() == 0 {
			// 计算这批的大小
			batchSize := uint64(GPUBatchSize)
			if batchSize > remaining {
				batchSize = remaining
			}

			startIndex := (j << 24) + offset

			// 在GPU上搜索
			ret := C.searchOnGPU(
				C.uint64_t(startIndex),
				C.uint64_t(batchSize),
				(*C.uint8_t)(unsafe.Pointer(&tarBytes[0])),
				(*C.uint8_t)(unsafe.Pointer(&result[0])),
				C.int(MaxThreadsPerThreadgroup),
			)

			if ret == 1 {
				// 找到了！
				foundFlag.Store(1)
				resultMutex.Lock()
				copy(foundResult[:], result)
				resultMutex.Unlock()
				fmt.Printf("\n[GPU Scheduler %d] 找到结果: %s\n", id, string(result))
				break
			}

			// 更新进度
			globalProgress.Add(int64(batchSize))

			offset += batchSize
			remaining -= batchSize

			// 检查上下文
			select {
			case <-ctx.Done():
				return
			default:
			}
		}
	}
}
