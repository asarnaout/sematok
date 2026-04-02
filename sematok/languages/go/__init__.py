"""
Go language configuration for sematok.

Covers .go files using the tree-sitter-go grammar. Patterns span the Go
ecosystem: web frameworks (Gin, Echo, Fiber, Chi), microservices (gRPC,
go-kit, Kratos), cloud infrastructure (Prometheus, Helm, Argo CD),
container runtimes (Docker/Moby, containerd), ORMs (GORM, Ent, sqlx),
CLI/TUI tools (Cobra, Bubble Tea), and standard library idioms.
"""

import re

import tree_sitter_go as tsgo
from tree_sitter import Language

from sematok.languages import LanguageConfig


def get_config() -> LanguageConfig:
    """Return the Go language configuration."""
    return LanguageConfig(
        name="go",
        file_extension=".go",
        tree_sitter_language=Language(tsgo.language()),
        unsafe_node_types=UNSAFE_NODE_TYPES,
        is_safe_override=None,  # Go doc comments are unstructured prose, no @param tags
        candidate_patterns=CANDIDATE_PATTERNS,
        seed_patterns=SEED_PATTERNS,
        subtree_root_types=SUBTREE_ROOT_TYPES,
        fixed_parent_types=FIXED_PARENT_TYPES,
        normalize_parent_types=NORMALIZE_PARENT_TYPES,
        structural_names=STRUCTURAL_NAMES,
        ident_pattern=r"([a-zA-Z_]\w*)",
        skip_path_patterns=SKIP_PATH_PATTERNS,
        repos=REPOS,
        eval_repos=EVAL_REPOS,
    )


# ---------------------------------------------------------------------------
# Lexer: safe zone detection
# ---------------------------------------------------------------------------

UNSAFE_NODE_TYPES = {
    "comment",                       # Both // and /* */ (single type in Go grammar)
    "interpreted_string_literal",    # "double-quoted" strings
    "raw_string_literal",            # `backtick` raw strings
    "rune_literal",                  # 'x' character literals
}

# ---------------------------------------------------------------------------
# Mining: regex candidate extraction
# ---------------------------------------------------------------------------

CANDIDATE_PATTERNS = [re.compile(p) for p in [
    # --- Imports ---
    r'import\s+"[^"]+"\s*$',
    r'import\s+\w+\s+"[^"]+"',

    # --- Error handling (Go's #1 boilerplate) ---
    r"if\s+err\s*!=\s*nil\s*\{",
    r"if\s+err\s*:=\s*\w+[^;{]*;\s*err\s*!=\s*nil\s*\{",
    r"return\s+(?:nil,\s*)?(?:fmt\.Errorf|errors\.(?:New|Wrap|Unwrap))\(",
    r"return\s+\w*,?\s*err\b",

    # --- Function / method declarations ---
    r"func\s+\(\s*\w+\s+\*?\w+\s*\)\s+\w+\s*\(",
    r"func\s+\w+\s*\(",

    # --- Struct / interface declarations ---
    r"type\s+\w+\s+struct\s*\{",
    r"type\s+\w+\s+interface\s*\{",

    # --- Struct tags ---
    r'`json:"[^"]*"`',
    r'`(?:json|yaml|xml|db|gorm|bson|mapstructure|toml|env):"[^"]*"',

    # --- Goroutines / defer ---
    r"go\s+func\s*\(",
    r"defer\s+(?:func\s*\(|[\w.]+\()",

    # --- Channel operations ---
    r"(?:make\(chan\s+|<-\s*\w+|select\s*\{)",
    r"case\s+\w+\s*(?::=\s*)?<-",

    # --- Context patterns ---
    r"ctx\s+context\.Context",
    r"context\.(?:Background|TODO|WithCancel|WithTimeout|WithDeadline|WithValue)\(",

    # --- Common stdlib calls ---
    r"fmt\.(?:Sprintf|Errorf|Printf|Println|Fprintf)\(",
    r"log\.(?:Fatal|Fatalf|Print|Printf|Println|Panic|Panicf)\(",
    r"os\.(?:Exit|Getenv|Open|Create|Remove|Stat|MkdirAll|ReadFile|WriteFile)\(",

    # --- HTTP patterns ---
    r"http\.(?:HandleFunc|Handle|ListenAndServe|Get|Post|NewRequest|Error)\(",
    r"func\s*\(\s*\w+\s+http\.ResponseWriter\s*,\s*\w+\s+\*http\.Request\s*\)",

    # --- Testing ---
    r"func\s+Test\w+\s*\(\s*t\s+\*testing\.T\s*\)",
    r"func\s+Benchmark\w+\s*\(\s*b\s+\*testing\.B\s*\)",
    r"t\.(?:Run|Error|Errorf|Fatal|Fatalf|Log|Logf|Skip|Helper|Cleanup|Parallel)\(",

    # --- Make / new ---
    r"make\(\s*(?:map|chan|\[\])\w*",

    # --- Sync primitives ---
    r"sync\.(?:Mutex|RWMutex|WaitGroup|Once|Map|Pool)\b",
]]

# ---------------------------------------------------------------------------
# Dictionary: seed patterns
# ---------------------------------------------------------------------------

SEED_PATTERNS = [
    # --- Error handling (12) -- Go's most iconic boilerplate ---
    ("if err != nil {", "error"),
    ("if err != nil {\n\t\treturn err\n\t}", "error"),
    ("if err != nil {\n\t\treturn nil, err\n\t}", "error"),
    ("return nil, err", "error"),
    ("return err", "error"),
    ("return fmt.Errorf(", "error"),
    ("errors.New(", "error"),
    ("errors.Is(", "error"),
    ("errors.As(", "error"),
    ("errors.Unwrap(", "error"),
    ("fmt.Errorf(\"", "error"),
    (", err := ", "error"),

    # --- Imports (10) ---
    ("\"context\"", "import"),
    ("\"fmt\"", "import"),
    ("\"errors\"", "import"),
    ("\"strings\"", "import"),
    ("\"strconv\"", "import"),
    ("\"sync\"", "import"),
    ("\"time\"", "import"),
    ("\"os\"", "import"),
    ("\"io\"", "import"),
    ("\"net/http\"", "import"),

    # --- Function / method patterns (6) ---
    ("func (", "function"),
    ("func main() {", "function"),
    ("func init() {", "function"),
    (") error {", "function"),
    (") (", "function"),
    ("interface{}", "function"),

    # --- Struct / type declarations (4) ---
    ("type ", "type"),
    (" struct {", "type"),
    (" interface {", "type"),
    ("var _ ", "type"),

    # --- Context (6) ---
    ("ctx context.Context", "context"),
    ("context.Background()", "context"),
    ("context.TODO()", "context"),
    ("context.WithCancel(", "context"),
    ("context.WithTimeout(", "context"),
    ("context.WithValue(", "context"),

    # --- Goroutines / concurrency (8) ---
    ("go func() {", "concurrency"),
    ("defer func() {", "concurrency"),
    ("<-ctx.Done()", "concurrency"),
    ("sync.Mutex", "concurrency"),
    ("sync.RWMutex", "concurrency"),
    ("sync.WaitGroup", "concurrency"),
    ("sync.Once", "concurrency"),
    ("make(chan ", "concurrency"),

    # --- Formatting / logging (8) ---
    ("fmt.Sprintf(", "format"),
    ("fmt.Errorf(", "format"),
    ("fmt.Printf(", "format"),
    ("fmt.Println(", "format"),
    ("fmt.Fprintf(", "format"),
    ("log.Fatal(", "logging"),
    ("log.Printf(", "logging"),
    ("log.Println(", "logging"),

    # --- HTTP (6) ---
    ("http.HandleFunc(", "http"),
    ("http.ListenAndServe(", "http"),
    ("http.Error(", "http"),
    ("http.StatusOK", "http"),
    ("http.StatusNotFound", "http"),
    ("http.StatusInternalServerError", "http"),

    # --- Testing (8) ---
    ("*testing.T", "test"),
    ("*testing.B", "test"),
    ("t.Run(", "test"),
    ("t.Errorf(", "test"),
    ("t.Fatalf(", "test"),
    ("t.Helper()", "test"),
    ("t.Parallel()", "test"),
    ("t.Cleanup(", "test"),

    # --- Common expressions (8) ---
    ("return nil", "expression"),
    ("!= nil", "expression"),
    ("== nil", "expression"),
    (":= make(", "expression"),
    (":= &", "expression"),
    (", ok := ", "expression"),
    (", ok)", "expression"),
    ("struct{}", "expression"),

    # --- Struct tags (4) ---
    ('`json:"', "tag"),
    ('`yaml:"', "tag"),
    (',omitempty"`', "tag"),
    ('`json:"-"`', "tag"),
]

# ---------------------------------------------------------------------------
# AST mining: subtree root types
# ---------------------------------------------------------------------------

SUBTREE_ROOT_TYPES = {
    # Statements
    "expression_statement", "short_var_declaration",
    "var_declaration", "const_declaration",
    "return_statement", "defer_statement", "go_statement",
    "if_statement", "for_statement",
    "expression_switch_statement", "type_switch_statement",
    "select_statement", "send_statement",
    "assignment_statement",
    # Declarations
    "function_declaration", "method_declaration",
    "type_declaration",
    # Expressions
    "call_expression",
    "import_declaration",
}

# ---------------------------------------------------------------------------
# Template mining: identifier normalization
# ---------------------------------------------------------------------------

# Parent node types where an identifier is structural (defines the pattern)
FIXED_PARENT_TYPES = {
    # Declarations
    "function_declaration", "method_declaration",
    "type_declaration", "type_spec",
    # Invocations & access
    "call_expression", "selector_expression",
    # Imports
    "import_spec", "import_declaration",
    # Types (use type_identifier / field_identifier, not identifier,
    # but included for completeness)
    "type_identifier", "field_identifier", "package_identifier",
    "qualified_type",
    "struct_type", "interface_type",
    # Interface compliance check: var _ Interface = (*Struct)(nil)
    "type_assertion_expression",
}

# Parent node types where an identifier is a user-chosen name (normalizable)
NORMALIZE_PARENT_TYPES = {
    # Variable declarations
    "short_var_declaration", "var_spec",
    # Assignments
    "assignment_statement",
    # Statements
    "return_statement", "defer_statement",
    # Arguments
    "argument_list",
    # Expressions
    "binary_expression", "unary_expression",
    "composite_literal",
    "index_expression", "slice_expression",
    # Loop targets
    "range_clause", "for_clause",
    # Parenthesized
    "parenthesized_expression",
}

# Well-known names that should never be normalized even if parent says so
STRUCTURAL_NAMES = {
    # --- Go keywords / builtins ---
    "nil", "true", "false", "iota", "_",
    "err", "ok",  # so common they're structural

    # --- Built-in types ---
    "bool", "string", "int", "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
    "float32", "float64", "complex64", "complex128",
    "byte", "rune", "error", "any", "comparable",

    # --- Built-in functions ---
    "make", "new", "len", "cap", "append", "copy", "delete",
    "close", "panic", "recover", "print", "println",
    "complex", "real", "imag", "min", "max", "clear",

    # --- Standard library package names ---
    "fmt", "os", "io", "net", "http", "context", "sync", "time",
    "strings", "strconv", "bytes", "errors", "log", "path",
    "filepath", "encoding", "json", "xml", "regexp", "sort",
    "math", "reflect", "testing", "flag", "bufio", "crypto",
    "database", "sql", "html", "template", "unicode", "runtime",
    "debug", "unsafe", "atomic", "signal", "exec", "embed",
    "slog", "cmp", "slices", "maps", "iter",
    "rand", "hash", "tar", "zip", "gzip",

    # --- Common stdlib types ---
    "Context", "Error", "Reader", "Writer", "Closer",
    "ReadCloser", "WriteCloser", "ReadWriter", "ReadWriteCloser",
    "Handler", "HandlerFunc", "Server", "Client",
    "Request", "Response", "ResponseWriter", "Header", "URL",
    "Conn", "Listener", "Addr",
    "Mutex", "RWMutex", "WaitGroup", "Once", "Pool", "Map", "Cond",
    "Group",  # errgroup.Group
    "Timer", "Ticker", "Duration", "Time", "Location",
    "Buffer", "Scanner",
    "Decoder", "Encoder", "Marshaler", "Unmarshaler",
    "Logger", "File", "FileInfo", "DirEntry",
    "Regexp", "Template",
    "Stringer", "Formatter",
    "Flag", "Value",
    "DB", "Tx", "Stmt", "Row", "Rows", "Result",

    # --- Common interface method names ---
    "String", "Close", "Read", "Write",
    "Len", "Less", "Swap", "ServeHTTP",
    "MarshalJSON", "UnmarshalJSON",
    "MarshalText", "UnmarshalText",
    "MarshalBinary", "UnmarshalBinary",

    # --- Common function / method names ---
    "Get", "Set", "Delete", "New", "Run", "Start", "Stop",
    "Init", "Open", "Create", "Remove", "Stat", "Walk",
    "Marshal", "Unmarshal", "Encode", "Decode", "Parse", "Format",
    "Handle", "HandleFunc", "ListenAndServe",
    "Sprintf", "Errorf", "Printf", "Println", "Fprintf",
    "Fatal", "Fatalf", "Panicf",
    "Background", "TODO", "WithCancel", "WithTimeout",
    "WithDeadline", "WithValue",
    "Lock", "Unlock", "RLock", "RUnlock",
    "Add", "Done", "Wait",
    "Do", "Load", "Store", "Range",

    # --- Testing names ---
    "T", "B", "M", "TB",
    "Logf", "Skipf",
    "Helper", "Cleanup", "TempDir", "Parallel",
    "Assert", "Require", "Suite",  # testify

    # --- Common variable names (structural in Go) ---
    "ctx", "mu", "wg",
    "req", "resp", "rw", "w", "r",

    # --- HTTP status constants ---
    "StatusOK", "StatusCreated", "StatusBadRequest",
    "StatusUnauthorized", "StatusForbidden", "StatusNotFound",
    "StatusInternalServerError", "StatusServiceUnavailable",
    "MethodGet", "MethodPost", "MethodPut", "MethodDelete",

    # --- Common framework names ---
    "Gin", "Echo", "Fiber", "Chi",
    "GORM", "Model", "Config",
    "Cobra", "Viper",
    "Zap", "SugaredLogger",
}

# ---------------------------------------------------------------------------
# Download: paths to skip when extracting source files
# ---------------------------------------------------------------------------

SKIP_PATH_PATTERNS = [
    "/vendor/",             # Go vendor directory
    ".pb.go",              # protobuf generated
    ".pb.gw.go",           # gRPC-gateway generated
    "_gen.go",             # code-generated files
    "_generated.go",
    "wire_gen.go",         # Wire DI generated
    "_easyjson.go",        # easyjson generated
    "_enumer.go",          # enumer generated
    "mock_",               # generated mock files
    "_mock.go",
    "/testdata/",          # test fixture data (non-Go files)
    "/third_party/",
    "zz_generated",        # Kubernetes-style generated
]

# ---------------------------------------------------------------------------
# Data: repos and eval splits
# ---------------------------------------------------------------------------

REPOS = [
    # --- Web frameworks & HTTP routers (4) ---
    ("gin-gonic", "gin"),                     # MIT          | Most popular Go web framework
    ("gofiber", "fiber"),                     # MIT          | Express-inspired, fasthttp-based
    ("labstack", "echo"),                     # MIT          | High-perf minimalist framework
    ("go-chi", "chi"),                        # MIT          | Lightweight idiomatic router

    # --- API & microservices (5) ---
    ("grpc", "grpc-go"),                      # Apache-2.0   | gRPC for Go
    ("go-kit", "kit"),                        # MIT          | Microservices toolkit
    ("go-kratos", "kratos"),                  # MIT          | Microservice framework (Bilibili)
    ("zeromicro", "go-zero"),                 # MIT          | Cloud-native microservices
    ("twitchtv", "twirp"),                    # Apache-2.0   | Simple RPC framework (Twitch)

    # --- Cloud infrastructure & DevOps (5) ---
    ("prometheus", "prometheus"),             # Apache-2.0   | Monitoring & alerting
    ("prometheus", "client_golang"),          # Apache-2.0   | Prometheus client library
    ("argoproj", "argo-cd"),                  # Apache-2.0   | GitOps continuous delivery
    ("pulumi", "pulumi"),                     # Apache-2.0   | Infrastructure as code
    ("dapr", "dapr"),                         # Apache-2.0   | Distributed app runtime

    # --- Container & orchestration (4) ---
    ("moby", "moby"),                         # Apache-2.0   | Docker engine
    ("containerd", "containerd"),             # Apache-2.0   | Container runtime
    ("helm", "helm"),                         # Apache-2.0   | Kubernetes package manager
    ("derailed", "k9s"),                      # Apache-2.0   | Kubernetes TUI dashboard

    # --- Database drivers & ORMs (5) ---
    ("go-gorm", "gorm"),                      # MIT          | Most popular Go ORM
    ("jmoiron", "sqlx"),                      # MIT          | Extensions to database/sql
    ("jackc", "pgx"),                         # MIT          | PostgreSQL driver
    ("redis", "go-redis"),                    # BSD-2-Clause | Redis client
    ("ent", "ent"),                           # Apache-2.0   | Entity framework (Meta)

    # --- CLI tools & TUI frameworks (4) ---
    ("spf13", "cobra"),                       # Apache-2.0   | CLI framework
    ("spf13", "viper"),                       # MIT          | Configuration management
    ("charmbracelet", "bubbletea"),           # MIT          | TUI framework
    ("charmbracelet", "lipgloss"),            # MIT          | Terminal styling

    # --- Networking & proxies (4) ---
    ("caddyserver", "caddy"),                 # Apache-2.0   | HTTP/2 web server
    ("traefik", "traefik"),                   # MIT          | Cloud-native reverse proxy
    ("coredns", "coredns"),                   # Apache-2.0   | DNS server
    ("valyala", "fasthttp"),                  # MIT          | High-perf HTTP

    # --- Messaging & streaming (3) ---
    ("nats-io", "nats-server"),               # Apache-2.0   | Cloud-native messaging
    ("nsqio", "nsq"),                         # MIT          | Realtime distributed messaging
    ("segmentio", "kafka-go"),                # MIT          | Kafka client

    # --- Cryptography & security (3) ---
    ("FiloSottile", "age"),                   # BSD-3-Clause | Simple file encryption
    ("cloudflare", "cfssl"),                  # BSD-2-Clause | PKI/TLS toolkit
    ("smallstep", "certificates"),            # Apache-2.0   | Private CA / ACME server

    # --- Testing & mocking (3) ---
    ("stretchr", "testify"),                  # MIT          | Test assertions & mocks
    ("onsi", "ginkgo"),                       # MIT          | BDD testing framework
    ("vektra", "mockery"),                    # BSD-3-Clause | Mock code generator

    # --- Logging & observability (3) ---
    ("uber-go", "zap"),                       # MIT          | Structured logging
    ("sirupsen", "logrus"),                   # MIT          | Structured logging (classic)
    ("open-telemetry", "opentelemetry-go"),   # Apache-2.0   | Distributed tracing

    # --- Data structures & utilities (2) ---
    ("emirpasic", "gods"),                    # BSD-2-Clause | Go data structures
    ("samber", "lo"),                         # MIT          | Lodash-style generics utility

    # --- Serialization & file formats (3) ---
    ("tidwall", "gjson"),                     # MIT          | Fast JSON parser
    ("json-iterator", "go"),                  # MIT          | High-perf JSON (drop-in)
    ("protocolbuffers", "protobuf-go"),       # BSD-3-Clause | Protocol Buffers for Go

    # --- Concurrency (3) ---
    ("sourcegraph", "conc"),                  # MIT          | Structured concurrency
    ("panjf2000", "ants"),                    # MIT          | Goroutine pool
    ("hibiken", "asynq"),                     # MIT          | Distributed task queue

    # --- Build tools & code generation (2) ---
    ("google", "wire"),                       # Apache-2.0   | Compile-time DI
    ("swaggo", "swag"),                       # MIT          | Swagger doc generator

    # --- Blockchain & distributed systems (1) ---
    ("hyperledger", "fabric"),                # Apache-2.0   | Enterprise blockchain

    # --- AI / ML (1) ---
    ("ollama", "ollama"),                     # MIT          | Local LLM runner

    # --- Image processing (1) ---
    ("disintegration", "imaging"),            # MIT          | Image processing library

    # --- Authentication & authorization (2) ---
    ("casbin", "casbin"),                     # Apache-2.0   | Authorization library
    ("golang-jwt", "jwt"),                    # MIT          | JWT implementation

    # --- WebSocket (1) ---
    ("gorilla", "websocket"),                 # BSD-2-Clause | WebSocket implementation
]

EVAL_REPOS = [
    "gofiber--fiber",                         # MIT          | Web framework
    "go-kratos--kratos",                      # MIT          | Microservices
    "argoproj--argo-cd",                      # Apache-2.0   | Cloud/GitOps
    "containerd--containerd",                 # Apache-2.0   | Container runtime
    "jackc--pgx",                             # MIT          | Database driver
    "charmbracelet--bubbletea",               # MIT          | TUI framework
    "nats-io--nats-server",                   # Apache-2.0   | Messaging
    "uber-go--zap",                           # MIT          | Logging
    "tidwall--gjson",                         # MIT          | Serialization
    "casbin--casbin",                         # Apache-2.0   | Authorization
]
