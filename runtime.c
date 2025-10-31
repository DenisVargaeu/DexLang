#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

void dex_print_string(const char *value) {
    if (!value) {
        puts("(null)");
        return;
    }
    puts(value);
}

void dex_print_number(double value) {
    printf("%.12g\n", value);
}

char *dex_input(const char *prompt) {
    if (!prompt) {
        prompt = "";
    }
    printf("%s", prompt);
    fflush(stdout);
    static char buf[256];
    if (!fgets(buf, sizeof(buf), stdin)) {
        return strdup("");
    }
    buf[strcspn(buf, "\n")] = 0;
    return strdup(buf);
}

char *dex_to_string(double n) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.12g", n);
    return strdup(buf);
}

double dex_to_number(const char *s) {
    if (!s) {
        return 0.0;
    }
    return atof(s);
}

int dex_len(const char *s) {
    if (!s) {
        return 0;
    }
    return (int)strlen(s);
}

char *dex_concat(const char *lhs, const char *rhs) {
    if (!lhs) {
        lhs = "";
    }
    if (!rhs) {
        rhs = "";
    }
    size_t left_len = strlen(lhs);
    size_t right_len = strlen(rhs);
    char *out = (char *)malloc(left_len + right_len + 1);
    if (!out) {
        return strdup("");
    }
    memcpy(out, lhs, left_len);
    memcpy(out + left_len, rhs, right_len);
    out[left_len + right_len] = '\0';
    return out;
}

char *dex_readfile(const char *path) {
    if (!path) {
        return strdup("");
    }
    FILE *handle = fopen(path, "rb");
    if (!handle) {
        return strdup("");
    }
    if (fseek(handle, 0, SEEK_END) != 0) {
        fclose(handle);
        return strdup("");
    }
    long length = ftell(handle);
    if (length < 0) {
        fclose(handle);
        return strdup("");
    }
    rewind(handle);
    char *buffer = (char *)malloc((size_t)length + 1);
    if (!buffer) {
        fclose(handle);
        return strdup("");
    }
    size_t read_bytes = fread(buffer, 1, (size_t)length, handle);
    buffer[read_bytes] = '\0';
    fclose(handle);
    return buffer;
}

double dex_writefile(const char *path, const char *text) {
    if (!path) {
        return 0.0;
    }
    if (!text) {
        text = "";
    }
    FILE *handle = fopen(path, "wb");
    if (!handle) {
        return 0.0;
    }
    size_t to_write = strlen(text);
    size_t written = fwrite(text, 1, to_write, handle);
    fclose(handle);
    return written == to_write ? 1.0 : 0.0;
}

double dex_sleep(double seconds) {
    if (seconds < 0) {
        seconds = 0;
    }
#ifdef _WIN32
    Sleep((DWORD)(seconds * 1000.0));
#else
    unsigned int usec = (unsigned int)(seconds * 1000000.0);
    if (usec > 0) {
        usleep(usec);
    }
#endif
    return 0.0;
}

double dex_exit(double code) {
    fflush(stdout);
    fflush(stderr);
    exit((int)code);
    return 0.0;
}

double dex_debug(const char *message) {
    if (!message) {
        message = "";
    }
    fprintf(stderr, "[debug] %s\n", message);
    fflush(stderr);
    return 0.0;
}

double dex_rand(double min_value, double max_value) {
    if (max_value < min_value) {
        double tmp = min_value;
        min_value = max_value;
        max_value = tmp;
    }
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    double span = max_value - min_value;
    double sample = rand() / (double)RAND_MAX;
    return min_value + sample * span;
}

double dex_system(const char *command) {
    if (!command) {
        return -1.0;
    }
    return (double)system(command);
}
