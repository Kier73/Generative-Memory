#include "../include/gmem_aro.h"
#include "gmem_internal.h"
#include <math.h>
#include <stdlib.h>


// --- AST Builder Helpers ---

static gmem_ast_node_t *gmem_ast_new(gmem_op_type_t type) {
  gmem_ast_node_t *node = (gmem_ast_node_t *)calloc(1, sizeof(gmem_ast_node_t));
  node->type = type;
  return node;
}

static gmem_ast_node_t *gmem_ast_const(float val) {
  gmem_ast_node_t *node = gmem_ast_new(GMEM_OP_CONST);
  node->val = val;
  return node;
}

static void gmem_ast_free(gmem_ast_node_t *node) {
  if (!node)
    return;
  gmem_ast_free(node->left);
  gmem_ast_free(node->right);
  free(node);
}

// --- Symbolic Simplifier ---

// Recursive optimizer
static gmem_ast_node_t *gmem_aro_optimize_node(gmem_ast_node_t *node) {
  if (!node)
    return NULL;

  // Optimize children first (Post-Order Traversal)
  node->left = gmem_aro_optimize_node(node->left);
  node->right = gmem_aro_optimize_node(node->right);

  // Constant Folding: If both children are constants, compute result
  if (node->left && node->right && node->left->type == GMEM_OP_CONST &&
      node->right->type == GMEM_OP_CONST) {

    float v1 = node->left->val;
    float v2 = node->right->val;
    float res = 0.0f;
    int folded = 1;

    switch (node->type) {
    case GMEM_OP_ADD:
      res = v1 + v2;
      break;
    case GMEM_OP_SUB:
      res = v1 - v2;
      break;
    case GMEM_OP_MUL:
      res = v1 * v2;
      break;
    case GMEM_OP_DIV:
      res = v2 != 0 ? v1 / v2 : 0;
      break; // Safe div
    default:
      folded = 0;
      break;
    }

    if (folded) {
      gmem_ast_free(node->left);
      gmem_ast_free(node->right);
      free(node);
      return gmem_ast_const(res);
    }
  }

  // Identity Rules
  // x * 1 -> x
  // x + 0 -> x
  if (node->type == GMEM_OP_MUL) {
    if (node->right && node->right->type == GMEM_OP_CONST &&
        node->right->val == 1.0f) {
      gmem_ast_node_t *l = node->left;
      free(node->right);
      free(node);
      return l;
    }
  }
  if (node->type == GMEM_OP_ADD) {
    if (node->right && node->right->type == GMEM_OP_CONST &&
        node->right->val == 0.0f) {
      gmem_ast_node_t *l = node->left;
      free(node->right);
      free(node);
      return l;
    }
  }

  // Null Rule
  // x * 0 -> 0
  if (node->type == GMEM_OP_MUL) {
    if ((node->right && node->right->type == GMEM_OP_CONST &&
         node->right->val == 0.0f) ||
        (node->left && node->left->type == GMEM_OP_CONST &&
         node->left->val == 0.0f)) {
      gmem_ast_free(node->left);
      gmem_ast_free(node->right);
      free(node);
      return gmem_ast_const(0.0f);
    }
  }

  return node;
}

// Main Simplification Entry Point
gmem_arr_rule_t gmem_aro_simplify(gmem_ctx_t ctx, uint64_t variety_sig,
                                  float *p1_out, float *p2_out) {
  // 1. Check existing registry (Cache)
  if (ctx && ctx->law_registry) {
    for (size_t i = 0; i < ctx->law_count; i++) {
      if (ctx->law_registry[i].variety_sig == variety_sig) {
        if (p1_out)
          *p1_out = ctx->law_registry[i].p1;
        if (p2_out)
          *p2_out = ctx->law_registry[i].p2;
        return ctx->law_registry[i].rule;
      }
    }
  }

  // 2. Symbolic Analysis (Hardened)
  // For this implementation, we map the Seed to an AST
  // Ideally this would come from a "Genetic Programming" engine or User Input
  // Here we hardcode the "Zero Seed" logic using the AST

  gmem_ast_node_t *root = NULL;

  if (variety_sig == 0) {
    // Construct AST: (x * 0) + 0
    // A standard "Null" generator
    root = gmem_ast_new(GMEM_OP_ADD);
    root->right = gmem_ast_const(0.0f);

    root->left = gmem_ast_new(GMEM_OP_MUL);
    root->left->left = gmem_ast_new(GMEM_OP_VAR_X);
    root->left->right = gmem_ast_const(0.0f); // * 0
  } else if (variety_sig == 0xDEADBEEF) {
    // Construct AST: x * 1.0 (Linear Identity)
    root = gmem_ast_new(GMEM_OP_MUL);
    root->left = gmem_ast_new(GMEM_OP_VAR_X);
    root->right = gmem_ast_const(1.0f);
  }

  // 3. Run Optimizer
  root = gmem_aro_optimize_node(root);

  // 4. Classify Result
  gmem_arr_rule_t rule = GMEM_ARR_NONE;
  float p1 = 0, p2 = 0;

  if (root) {
    if (root->type == GMEM_OP_CONST) {
      rule = GMEM_ARR_CONST;
      p1 = root->val;
    } else if (root->type == GMEM_OP_VAR_X) {
      rule = GMEM_ARR_LINEAR;
      p1 = 0; // b
      p2 = 1; // m
    }
    // Check for Linear Form (mx + b) - unimplemented for complex trees
  }

  gmem_ast_free(root);

  // 5. Register Result
  if (rule != GMEM_ARR_NONE) {
    gmem_law_register(ctx, variety_sig, rule, p1, p2);
    if (p1_out)
      *p1_out = p1;
    if (p2_out)
      *p2_out = p2;
  }

  return rule;
}

void gmem_law_register(gmem_ctx_t ctx, uint64_t variety_sig,
                       gmem_arr_rule_t rule, float p1, float p2) {
  if (!ctx)
    return;

  // Check if exists
  for (size_t i = 0; i < ctx->law_count; i++) {
    if (ctx->law_registry[i].variety_sig == variety_sig) {
      ctx->law_registry[i].rule = rule;
      ctx->law_registry[i].p1 = p1;
      ctx->law_registry[i].p2 = p2;
      return;
    }
  }

  ctx->law_count++;
  ctx->law_registry = (gmem_law_entry_t *)realloc(
      ctx->law_registry, ctx->law_count * sizeof(gmem_law_entry_t));

  ctx->law_registry[ctx->law_count - 1].variety_sig = variety_sig;
  ctx->law_registry[ctx->law_count - 1].rule = rule;
  ctx->law_registry[ctx->law_count - 1].p1 = p1;
  ctx->law_registry[ctx->law_count - 1].p2 = p2;
}
