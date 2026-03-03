#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

#define INCX 1
#define INCY 1

typedef struct {
    int total;
    int pass;
    int fail;
} TestStats;

TestStats stats;
int test_number = 0;

void print_test_result(const char* test_name, int pass) {
    stats.total++;
    test_number++;
    
    if (pass) {
        stats.pass++;
        printf("   Тест #%02d %-20s ... OK\n", test_number, test_name);
    } else {
        stats.fail++;
        printf("   Тест #%02d %-20s ... FAIL\n", test_number, test_name);
    }
}

void test_sgemv() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 0, 0, 1.0f, A, 1, x, INCX, 0.0f, y, INCY);
    print_test_result("sgemv", pass);
}

void test_dgemv() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 0, 0, 1.0, A, 1, x, INCX, 0.0, y, INCY);
    print_test_result("dgemv", pass);
}

void test_cgemv() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    float beta[2] = {0.0f, 0.0f};
    cblas_cgemv(CblasRowMajor, CblasNoTrans, 0, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    print_test_result("cgemv", pass);
}

void test_zgemv() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans, 0, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    print_test_result("zgemv", pass);
}

void test_ssymv() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_ssymv(CblasRowMajor, CblasUpper, 0, 1.0f, A, 1, x, INCX, 0.0f, y, INCY);
    print_test_result("ssymv", pass);
}

void test_dsymv() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dsymv(CblasRowMajor, CblasLower, 0, 1.0, A, 1, x, INCX, 0.0, y, INCY);
    print_test_result("dsymv", pass);
}

void test_chemv() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    float beta[2] = {0.0f, 0.0f};
    cblas_chemv(CblasRowMajor, CblasUpper, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    print_test_result("chemv", pass);
}

void test_zhemv() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    cblas_zhemv(CblasRowMajor, CblasUpper, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    print_test_result("zhemv", pass);
}

void test_strmv() {
    float A[1], x[1];
    int pass = 1;
    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    print_test_result("strmv", pass);
}

void test_dtrmv() {
    double A[1], x[1];
    int pass = 1;
    cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    print_test_result("dtrmv", pass);
}

void test_ctrmv() {
    float A[2], x[2];
    int pass = 1;
    cblas_ctrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    print_test_result("ctrmv", pass);
}

void test_ztrmv() {
    double A[2], x[2];
    int pass = 1;
    cblas_ztrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    print_test_result("ztrmv", pass);
}

void test_strsv() {
    float A[1], x[1];
    int pass = 1;
    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    print_test_result("strsv", pass);
}

void test_dtrsv() {
    double A[1], x[1];
    int pass = 1;
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, 0, A, 1, x, INCX);
    print_test_result("dtrsv", pass);
}

void test_ctrsv() {
    float A[2], x[2];
    int pass = 1;
    cblas_ctrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    print_test_result("ctrsv", pass);
}

void test_ztrsv() {
    double A[2], x[2];
    int pass = 1;
    cblas_ztrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, 0, A, 1, x, INCX);
    print_test_result("ztrsv", pass);
}

void test_sger() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_sger(CblasRowMajor, 0, 0, 1.0f, x, INCX, y, INCY, A, 1);
    print_test_result("sger", pass);
}

void test_dger() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dger(CblasRowMajor, 0, 0, 1.0, x, INCX, y, INCY, A, 1);
    print_test_result("dger", pass);
}

void test_cgeru() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    cblas_cgeru(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    print_test_result("cgeru", pass);
}

void test_zgeru() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    cblas_zgeru(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    print_test_result("zgeru", pass);
}

void test_cgerc() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    cblas_cgerc(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    print_test_result("cgerc", pass);
}

void test_zgerc() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    cblas_zgerc(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    print_test_result("zgerc", pass);
}

void test_ssyr() {
    float A[1], x[1];
    int pass = 1;
    cblas_ssyr(CblasRowMajor, CblasLower, 0, 1.0f, x, INCX, A, 1);
    print_test_result("ssyr", pass);
}

void test_dsyr() {
    double A[1], x[1];
    int pass = 1;
    cblas_dsyr(CblasRowMajor, CblasUpper, 0, 1.0, x, INCX, A, 1);
    print_test_result("dsyr", pass);
}

void test_cher() {
    float A[2], x[2];
    int pass = 1;
    float alpha = 1.0f;
    cblas_cher(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, A, 1);
    print_test_result("cher", pass);
}

void test_zher() {
    double A[2], x[2];
    int pass = 1;
    double alpha = 1.0;
    cblas_zher(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, A, 1);
    print_test_result("zher", pass);
}

void test_ssyr2() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_ssyr2(CblasRowMajor, CblasLower, 0, 1.0f, x, INCX, y, INCY, A, 1);
    print_test_result("ssyr2", pass);
}

void test_dsyr2() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dsyr2(CblasRowMajor, CblasUpper, 0, 1.0, x, INCX, y, INCY, A, 1);
    print_test_result("dsyr2", pass);
}

void test_cher2() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    cblas_cher2(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, y, INCY, A, 1);
    print_test_result("cher2", pass);
}

void test_zher2() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    cblas_zher2(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, y, INCY, A, 1);
    print_test_result("zher2", pass);
}

int main() {
    printf("\n");
    printf(" ____________________________________________\n");
    printf("|     CBLAS Level 2 — Проверка функций       |\n");
    printf("|____________________________________________|\n\n");
    
    stats.total = 0;
    stats.pass = 0;
    stats.fail = 0;
    test_number = 0;
    
    printf(" Группа 1: GEMV (General Matrix-Vector)\n");
    printf("─────────────────────────────────────────\n");
    test_sgemv();
    test_dgemv();
    test_cgemv();
    test_zgemv();
    printf("\n");
    
    printf(" Группа 2: SYMV/HEMV (Symmetric/Hermitian)\n");
    printf("────────────────────────────────────────────\n");
    test_ssymv();
    test_dsymv();
    test_chemv();
    test_zhemv();
    printf("\n");
    
    printf(" Группа 3: TRMV (Triangular Matrix-Vector)\n");
    printf("───────────────────────────────────────────\n");
    test_strmv();
    test_dtrmv();
    test_ctrmv();
    test_ztrmv();
    printf("\n");
    
    printf(" Группа 4: TRSV (Triangular Solve)\n");
    printf("───────────────────────────────────\n");
    test_strsv();
    test_dtrsv();
    test_ctrsv();
    test_ztrsv();
    printf("\n");
    
    printf(" Группа 5: GER (General Rank-1 Update)\n");
    printf("───────────────────────────────────────\n");
    test_sger();
    test_dger();
    test_cgeru();
    test_zgeru();
    test_cgerc();
    test_zgerc();
    printf("\n");
    
    printf(" Группа 6: SYR/HER (Symmetric/Hermitian Rank-1)\n");
    printf("────────────────────────────────────────────────\n");
    test_ssyr();
    test_dsyr();
    test_cher();
    test_zher();
    printf("\n");
    
    printf(" Группа 7: SYR2/HER2 (Rank-2 Update)\n");
    printf("─────────────────────────────────────\n");
    test_ssyr2();
    test_dsyr2();
    test_cher2();
    test_zher2();
    printf("\n");
    
    printf(" ____________________________________________\n");
    printf("|              ИТОГОВЫЙ ОТЧЁТ                |\n");
    printf("|____________________________________________|\n");
    printf("|  Всего выполнено тестов: %-18d |\n", stats.total);
    printf("|  Успешно завершено:      %-18d |\n", stats.pass);
    printf("|  Ошибок обнаружено:      %-18d |\n", stats.fail);
    printf(" ________________________________|\n");
    
    if (stats.fail == 0) {
        printf("|  СТАТУС: ВСЕ ТЕСТЫ ПРОЙДЕНЫ             |\n");
    } else {
        printf("|  СТАТУС: ТРЕБУЕТСЯ ВНИМАНИЕ             |\n");
    }
    printf("______________________________________________\n\n");
    
    return 0;
}