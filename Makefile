CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm

# SIMD flags - auto-detect
SIMD_FLAGS := $(shell $(CC) -mavx2 -E -x c /dev/null >/dev/null 2>&1 && echo "-mavx2 -mfma" || ($(CC) -msse2 -E -x c /dev/null >/dev/null 2>&1 && echo "-msse2"))
CFLAGS += $(SIMD_FLAGS)

SRCDIR = src
INCDIR = include
BUILDDIR = build

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))

.PHONY: all clean test

all: $(BUILDDIR) $(BUILDDIR)/libdeeplearn.a $(BUILDDIR)/test_tensor $(BUILDDIR)/train_gpt2 $(BUILDDIR)/inference

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/libdeeplearn.a: $(OBJECTS)
	ar rcs $@ $^

$(BUILDDIR)/test_tensor: tests/test_tensor.c $(BUILDDIR)/libdeeplearn.a
	$(CC) $(CFLAGS) $< -L$(BUILDDIR) -ldeeplearn $(LDFLAGS) -o $@

$(BUILDDIR)/train_gpt2: examples/train_gpt2.c $(BUILDDIR)/libdeeplearn.a
	$(CC) $(CFLAGS) $< -L$(BUILDDIR) -ldeeplearn $(LDFLAGS) -o $@

$(BUILDDIR)/inference: examples/inference.c $(BUILDDIR)/libdeeplearn.a
	$(CC) $(CFLAGS) $< -L$(BUILDDIR) -ldeeplearn $(LDFLAGS) -o $@

test: $(BUILDDIR)/test_tensor
	./$(BUILDDIR)/test_tensor

clean:
	rm -rf $(BUILDDIR)
