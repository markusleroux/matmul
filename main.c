#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define DISABLE_NEON

#if defined(__ARM_NEON) && !defined(DISABLE_NEON)
#define NEON_SUPPORT 1
#else
#define NEON_SUPPORT 0
#endif

#if !defined(__ARM_64BIT_STATE) || !defined(__ARM_FEATURE_QRDMX)
#define QRDMX_SUPPORT 0
#else
#define QRDMX_SUPPORT 1
#endif

#if NEON_SUPPORT
#include <arm_neon.h>
#endif

#if !NEON_SUPPORT

void matrix_vector_multiplication(const int8_t *matrix, // column major
                                  int32_t num_rows, int32_t num_columns,
                                  const int16_t *input, int16_t *output) {
  /* Computes the product of matrix by a vector using either
   * a 32 bit or 16 bit intermediary */
  int64_t i = 0;
  for (int32_t i_x = 0, i_y = 0; i < (int64_t)num_rows * num_columns;
       ++i, ++i_y) {
    if (i_y == num_rows) {
      i_y = 0;
      ++i_x;
    }

#ifdef __32BIT_PRODUCT
    output[i_y] += (int32_t)matrix[i] * input[i_x] >> 7;
#else
    // uses 16 bit intermediary
    output[i_y] += (matrix[i] * (input[i_x] & ((1 << 8) - 1))) >> 7;
    output[i_y] += (matrix[i] * (input[i_x] >> 8)) << 1;
#endif
  }
}

#elif defined(HIGH_PRECISION)

static inline void product(const int8_t *matrix, // 8x1
                           int16_t v,            // 1x1
                           int32_t *result       // 8x1
) {
  /* Computes the product using vmlal and a 32 bit intermediate result. */
  int32x4x2_t result32x4x2 = vld2q_s32(result);
  int16x8_t matrix_col16x8 = vmovl_s8(vld1_s8(matrix)); // load into 16 bits

  result32x4x2.val[0] =
      vmlal_n_s16(result32x4x2.val[0], vget_low_s16(matrix_col16x8), v);
  result32x4x2.val[1] =
      vmlal_n_s16(result32x4x2.val[1], vget_high_s16(matrix_col16x8), v);

  vst2q_s32(result, result32x4x2);
}

static void squash(int32_t *result, int16_t *output, int32_t num_rows) {
  /* Takes each element in result, applies a saturating rounding narrowing right
   * shift and stores the result in output */
  int32_t i_y = num_rows - 8;
  for (; i_y > -1; i_y -= 8) {
    int32x4x2_t result32x4x2 = vld2q_s32(result + i_y);

    // saturate, round, narrow
    vst1_s16(output + i_y, vqrshrn_n_s32(result32x4x2.val[0], 7));
    vst1_s16(output + i_y + 4, vqrshrn_n_s32(result32x4x2.val[1], 7));
  }

  // last 7 entries
  for (int8_t i = 0; i < 8 + i_y; ++i) {
    int32x4_t result32x4 = vld1q_lane_s32(result + i, vdupq_n_s32(0), 0);
    vst1_lane_s16(output + i, vqrshrn_n_s32(result32x4, 7), 0);
  }
}

void matrix_vector_multiplication(const int8_t *matrix, // column major
                                  int32_t num_rows, int32_t num_columns,
                                  const int16_t *input, int16_t *output) {
  /* Computes the product of a matrix by a vector using a 32 bit intermediary.
   * More accurate, but requires more time and quite a bit of space */
  int32_t *result = (int32_t *)malloc(num_rows * 4);
  int64_t index = (int64_t)num_columns * num_rows - 8;
  for (int32_t i_x = num_columns - 1, i_y; i_x > -1; --i_x) {
    for (i_y = num_rows - 8; i_y > -1; i_y -= 8, index -= 8) {
      product(matrix + index, input[i_x], result + i_y);
    }
    for (i_y += 7, index += 7; i_y > -1; --i_y, --index) {
      result[i_y] += (int32_t)matrix[index] * input[i_x];
    }
  }

  squash(result, output, num_rows);
  free(result);
}

#elif !QRDMX_SUPPORT

static inline void product(const int8_t *matrix,  // 8x1
                           const int16_t *vector, // 1x1
                           int16_t *output        // 8x1
) {
  /* Computes the product of the first 8 elements of matrix by the first element
   * of vector and accumulates the result into output */
  int16x8_t result16x8 = vld1q_s16(output);
  int8x8_t matrix_col8x8 = vld1_s8(matrix);

  // vector widening shift left by constant
  int16x8_t matrix_col16x8 = vshll_n_s8(matrix_col8x8, 8);

  // vector saturating doubling multiply by scalar on q register and get high
  // half
  matrix_col16x8 = vqdmulhq_n_s16(matrix_col16x8, *vector);

  // matrix_col16x8 is << 8, so matrix_col16x8 * v has 8 trailing 0. it is
  // doubled once so we effectively remove 7 digits by selecting the high half -
  // the point stays fixed
  result16x8 = vqaddq_s16(matrix_col16x8,
                          result16x8); // vector saturating addition on quad

  vst1q_s16(output, result16x8);
}

static inline void
product_nrow(const int8_t *matrix,  // pointer to first element in row
             const int16_t *vector, // pointer to corresponding vector element
             int16_t *output,       // pointer to first element in output
             int8_t n // number of rows to process, 0 < n <= min(8, num_rows)
) {
  /* Computes the product of the first n elements of matrix by the first element
   * of vector and accumulates the result into output */
  for (int i = 0; i < n; ++i) {
    int16x8_t result16x8 = vld1q_lane_s16(output + i, vdupq_n_s16(0), 0);
    int8x8_t matrix_col8x8 = vld1_lane_s8(matrix + i, vdup_n_s8(0), 0);

    // vector widening shift left by constant
    int16x8_t matrix_col16x8 = vshll_n_s8(matrix_col8x8, 8);

    // vector saturating doubling multiply by scalar on q register and get high
    // half
    matrix_col16x8 = vqdmulhq_n_s16(matrix_col16x8, *vector);

    // matrix_col16x8 is << 8, so matrix_col16x8 * v has 8 trailing 0. it is
    // doubled once so we effectively remove 7 digits by selecting the high half
    // - the point stays fixed
    result16x8 = vqaddq_s16(matrix_col16x8,
                            result16x8); // vector saturating addition on quad

    vst1q_lane_s16(output + i, result16x8, 0);
  }
}

void matrix_vector_multiplication(const int8_t *matrix, // column major
                                  int32_t num_rows, int32_t num_columns,
                                  const int16_t *input, int16_t *output) {
  /* Computes the product of a matrix by a vector using a left shift, the
   * vqdmulhq_n_s16 intrinsic and a saturating add. */
  int64_t index = (int64_t)num_columns * num_rows - 8;
  for (int32_t i_x = num_columns - 1, i_y; i_x > -1; --i_x, index -= 8) {
    for (i_y = num_rows - 8; i_y > -1; i_y -= 8, index -= 8) {
      product(matrix + index, input + i_x, output + i_y);
    }

    // index moves to first in column
    index -= i_y;
    // handle last 7 or fewer elements individually
    product_nrow(matrix + index, input + i_x, output, 8 + i_y);
  }
}

#else // only available on armv8.1 (not rpi3)

static inline void product(const int8_t *matrix, int16x4_t vector16x4,
                           int16_t *output, int32_t num_rows) {
  /* Computes the product of the first 8 elements of 4 rows of the matrix by the
   * first 4 element of vector and accumulates the result into output using the
   * vqrdmlahq_laneq_s16 intrinsic. Not available on pi. */
  int16x8_t result16x8 = vld1q_s16(output);

  // the following 2 operations could probably be 1 in assembly
  // load byte into second half of 16 bit. Would mean you could pack
  // registers tighter
  int16x8x4_t matrix_col16x8x4 = {
      vshll_n_s8(vld1_s8(matrix), 8),
      vshll_n_s8(vld1_s8(matrix + num_rows), 8),
      vshll_n_s8(vld1_s8(matrix + 2 * num_rows), 8),
      vshll_n_s8(vld1_s8(matrix + 3 * num_rows), 8),
  };

  result16x8 =
      vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[0], vector16x4, 0);
  result16x8 =
      vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[1], vector16x4, 1);
  result16x8 =
      vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[2], vector16x4, 2);
  result16x8 =
      vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[3], vector16x4, 3);

  vst1q_s16(output, result16x8);
}

static inline void
product_nrow(const int8_t *matrix, // pointer to first element in first row
             int16x4_t vector16x4, // 4 elements from input vector
             int16_t *output,      // pointer to first element in output
             int32_t num_rows,
             int8_t n // number of rows to process, 0 < n <= min(8, num_rows)
) {
  /* Computes the product of the first n elements of 4 rows of the matrix by the
   * first 4 element of vector and accumulates the result into output using the
   * vqrdmlahq_laneq_s16 intrinsic. Not available on pi. */
  for (int i = 0; i < n; ++i) {
    int16x8_t result16x8 = vld1q_lane_s16(output + i, vdupq_n_s16(0), 0);

    // the following 2 operations could probably be 1 in assembly
    // load byte into second half of 16 bit. Would mean you could pack
    // registers tighter
    int16x8x4_t matrix_col16x8x4 = {
        vshll_n_s8(vld1_lane_s8(matrix + i, vdup_n_s8(0), 0), 8),
        vshll_n_s8(vld1_lane_s8(matrix + num_rows + i, vdup_n_s8(0), 0), 8),
        vshll_n_s8(vld1_lane_s8(matrix + 2 * num_rows + i, vdup_n_s8(0), 0), 8),
        vshll_n_s8(vld1_lane_s8(matrix + 3 * num_rows + i, vdup_n_s8(0), 0), 8),
    };

    result16x8 =
        vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[0], vector16x4, 0);
    result16x8 =
        vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[1], vector16x4, 1);
    result16x8 =
        vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[2], vector16x4, 2);
    result16x8 =
        vqrdmlahq_lane_s16(result16x8, matrix_col16x8x4.val[3], vector16x4, 3);

    vst1q_lane_s16(output + i, result16x8, 0);
  }
}

static inline void product_nrow_1col(
    const int8_t *matrix, // pointer to first element in first row
    int16_t v,            // the value to multiply by
    int16_t *output,      // pointer to first element in output
    int8_t n // number of rows to process, 0 < n <= min(8, num_rows)
) {
  /* Computes the product of the first n elements of 1 row of the matrix by the
   * first element of vector and accumulates the result into output using the
   * vqrdmlahq_laneq_s16 intrinsic. Not available on pi. */

  // workaround for missing vqrdmlahq_n
  int16x4_t vector16x4 = vld1_lane_s16(&v, vdup_n_s16(0), 0);

  for (int i = 0; i < n; ++i) {
    int16x8_t result16x8 = vld1q_lane_s16(output + i, vdupq_n_s16(0), 0);
    int16x8_t matrix_col16x8 =
        vshll_n_s8(vld1_lane_s8(matrix + i, vdup_n_s8(0), 0), 8);
    result16x8 = vqrdmlahq_lane_s16(result16x8, matrix_col16x8, vector16x4, 0);
    vst1q_lane_s16(output + i, result16x8, 0);
  }
}

void matrix_vector_multiplication(const int8_t *matrix, // column major
                                  int32_t num_rows, int32_t num_columns,
                                  const int16_t *vector, int16_t *output) {
  // set assocative cache is probably ok to access 4 indices (need to test)
  int32_t i_x = num_columns - 4;
  for (int32_t i_y; i_x > -1; i_x -= 4) {
    int16x4_t vector16x4 = vld1_s16(vector + i_x);
    int64_t i_base =
        ((int64_t)i_x + 1) * num_rows - 8; // eight last element in column i_x

    for (i_y = num_rows - 8; i_y > -1; i_y -= 8) {
      product(matrix + i_base, vector16x4, output + i_y, num_rows);
    }

    // remaining rows
    product_nrow(matrix + (int64_t)i_x * num_rows, vector16x4, output, num_rows,
                 8 - i_y);
  }

  // first three or fewer columns
  i_x += 3; // i_x is last column not seen
  for (int64_t index = ((int64_t)i_x + 1) * num_rows - 8; i_x > -1;
       --i_x, index -= 8) {
    int32_t i_y = num_rows - 1;
    for (; i_y > -1; i_y -= 8, index -= 8) { // could iterate 8 at a time too
      product_nrow_1col(matrix + index, vector[i_x], output + i_y, 8);
    }
    index -= i_y; // index in first position of row
    product_nrow_1col(matrix + index, vector[i_x], output, 8 - i_y);
  }
}

#endif

long int
perf_test(void (*matrix_vector_multiplication)(const int8_t *, int32_t, int32_t,
                                               const int16_t *, int16_t *),
          const int8_t *matrix, int32_t num_rows, int32_t num_columns,
          const int16_t *input, clockid_t clk_id) {
  /* Use clock_gettime to measure either wall time or cpu time */
  int16_t *output = (int16_t *)malloc(2 * num_rows);

  struct timespec start, end;
  clock_gettime(clk_id, &start);
  matrix_vector_multiplication(matrix, num_rows, num_columns, input, output);
  clock_gettime(clk_id, &end);

  free(output);
  return (end.tv_sec - start.tv_sec) * (long)1e9 +
         (end.tv_nsec - start.tv_nsec);
}

long int
real_time(void (*matrix_vector_multiplication)(const int8_t *, int32_t, int32_t,
                                               const int16_t *, int16_t *),
          const int8_t *matrix, int32_t num_rows, int32_t num_columns,
          const int16_t *input) {
  return perf_test(matrix_vector_multiplication, matrix, num_rows, num_columns,
                   input, CLOCK_REALTIME);
}

long int cpu_time(void (*matrix_vector_multiplication)(const int8_t *, int32_t,
                                                       int32_t, const int16_t *,
                                                       int16_t *),
                  const int8_t *matrix, int32_t num_rows, int32_t num_columns,
                  const int16_t *input) {
  return perf_test(matrix_vector_multiplication, matrix, num_rows, num_columns,
                   input, CLOCK_PROCESS_CPUTIME_ID);
}

// allocates on the heap and doesn't free!
int8_t *matrix_gen(int32_t num_rows, int32_t num_cols) {
  int8_t *matrix = (int8_t *)malloc(num_rows * num_cols);
  for (int64_t i = 0; i < num_rows * num_cols; ++i) {
    matrix[i] = random();
  }
  return matrix;
}

// allocates on the heap and doesn't free!
int16_t *vector_gen(int32_t num_rows) {
  int16_t *vector = (int16_t *)malloc(2 * num_rows);
  for (int64_t i = 0; i < num_rows; ++i) {
    vector[i] = random();
  }
  return vector;
}

#define UTF_CHECK "\xE2\x9C\x93\n"
#define UTF_X "\xE2\x9C\x95\n"

void print_available_intrinsics() {
  printf("Compiled with support for:\n");
  printf("  NEON:  %s", NEON_SUPPORT ? UTF_CHECK : UTF_X);
  printf("  QRDMX: %s\n", QRDMX_SUPPORT ? UTF_CHECK : UTF_X);
}

#ifndef ITERATIONS
#define ITERATIONS 100
#endif

int main(int argc, char *argv[]) {
  print_available_intrinsics();

  if (argc <= 2) {
    printf("Expected two integer inputs m and n, representing dimensions of m "
           "x n matrix\n");
    exit(1);
  }
  int m = atoi(argv[1]), n = atoi(argv[2]);

  int8_t *big_matrix = matrix_gen(m, n);
  int16_t *big_vector = vector_gen(n);

  time_t cpu_ttime = 0;
  for (int i = 0; i < ITERATIONS; ++i) {
    cpu_ttime +=
        cpu_time(matrix_vector_multiplication, big_matrix, m, n, big_vector);
  }

  time_t wall_ttime = 0;
  for (int i = 0; i < ITERATIONS; ++i) {
    wall_ttime +=
        real_time(matrix_vector_multiplication, big_matrix, m, n, big_vector);
  }
  printf("CPU time: %ld | Wall time: %ld\n", cpu_ttime / ITERATIONS,
         wall_ttime / ITERATIONS);

  free(big_matrix);
  free(big_vector);
}
