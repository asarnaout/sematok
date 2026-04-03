"""Java language configuration for sematok."""

import re

import tree_sitter_java as tsjava
from tree_sitter import Language

from sematok.languages import LanguageConfig


def _is_javadoc(node, source_bytes: bytes) -> bool:
    """Treat Javadoc comments as safe (compressible).

    A Javadoc comment starts with ``/**`` and contains highly repetitive
    boilerplate (@param, @return, @throws, @see, {@link ...}, {@code ...}).
    Regular block comments (``/* ... */``) remain unsafe.
    """
    if node.type != "block_comment":
        return False
    text = source_bytes[node.start_byte:node.end_byte]
    return text.startswith(b"/**")


def get_config() -> LanguageConfig:
    """Return the Java language configuration."""
    return LanguageConfig(
        name="java",
        file_extension=".java",
        tree_sitter_language=Language(tsjava.language()),
        unsafe_node_types=UNSAFE_NODE_TYPES,
        is_safe_override=_is_javadoc,
        candidate_patterns=CANDIDATE_PATTERNS,
        seed_patterns=SEED_PATTERNS,
        subtree_root_types=SUBTREE_ROOT_TYPES,
        fixed_parent_types=FIXED_PARENT_TYPES,
        normalize_parent_types=NORMALIZE_PARENT_TYPES,
        structural_names=STRUCTURAL_NAMES,
        ambiguous_parent_types={"formal_parameter"},
        ident_pattern=r"([a-zA-Z_$][\w$]*)",
        skip_path_patterns=SKIP_PATH_PATTERNS,
        repos=REPOS,
        eval_repos=EVAL_REPOS,
    )


# ---------------------------------------------------------------------------
# Lexer: safe zone detection
# ---------------------------------------------------------------------------

UNSAFE_NODE_TYPES = {
    "line_comment",         # // comments
    "block_comment",        # /* ... */ and /** ... */ (Javadoc overridden as safe)
    "string_literal",       # "..." strings and """...""" text blocks
    "character_literal",    # '.' character literals
}

# ---------------------------------------------------------------------------
# Mining: regex candidate extraction
# ---------------------------------------------------------------------------

CANDIDATE_PATTERNS = [
    # Import patterns
    re.compile(r"import\s+(?:static\s+)?[\w.]+(?:\.\*)?;"),

    # Annotation patterns
    re.compile(
        r"@(?:Override|Deprecated|SuppressWarnings|FunctionalInterface"
        r"|SafeVarargs|Nullable|NonNull|NotNull)"
    ),
    re.compile(
        r"@(?:Test|Before|After|BeforeEach|AfterEach|BeforeAll|AfterAll"
        r"|ParameterizedTest|DisplayName|Disabled|Timeout"
        r"|ExtendWith|RunWith|Mock|InjectMocks|Captor|Spy)"
        r"(?:\([^)\n]*\))?"
    ),
    re.compile(
        r"@(?:Autowired|Inject|Bean|Component|Service|Repository"
        r"|Controller|RestController|Configuration|Qualifier"
        r"|Value|Scope|Lazy|Primary|ConditionalOn\w+)"
        r"(?:\([^)\n]*\))?"
    ),
    re.compile(
        r"@(?:RequestMapping|GetMapping|PostMapping|PutMapping"
        r"|DeleteMapping|PatchMapping|PathVariable|RequestParam"
        r"|RequestBody|ResponseBody|ResponseStatus)"
        r"(?:\([^)\n]*\))?"
    ),
    re.compile(
        r"@(?:Entity|Table|Column|Id|GeneratedValue|ManyToOne"
        r"|OneToMany|ManyToMany|JoinColumn|Transactional"
        r"|Cacheable|CacheEvict)\b"
        r"(?:\([^)\n]*\))?"
    ),

    # Access modifier combos
    re.compile(
        r"(?:public|private|protected)\s+"
        r"(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:synchronized\s+)?"
        r"(?:void|int|long|boolean|String|Object|List|Map|Set|Optional)"
    ),

    # Common method signatures
    re.compile(r"public\s+static\s+void\s+main\s*\(\s*String\s*\[\s*\]\s+\w+\s*\)"),
    re.compile(
        r"(?:public|private|protected)\s+\w+\s+(?:get|set|is|has|can|should|with)\w*\s*\("
    ),

    # Exception patterns
    re.compile(
        r"throw\s+new\s+(?:IllegalArgumentException|IllegalStateException"
        r"|NullPointerException|UnsupportedOperationException"
        r"|RuntimeException|IOException|ClassNotFoundException"
        r"|NoSuchElementException|IndexOutOfBoundsException)\s*\("
    ),
    re.compile(r"catch\s*\(\s*\w+(?:\s*\|\s*\w+)*\s+\w+\s*\)"),

    # Generic type patterns
    re.compile(r"<\s*\w+\s+extends\s+[^>]+>"),
    re.compile(
        r"(?:List|Map|Set|Queue|Deque|Collection|Optional|Stream"
        r"|Iterable|Iterator|Supplier|Consumer|Function|Predicate"
        r"|CompletableFuture|Future|Callable)\s*<"
    ),

    # Stream / functional patterns
    re.compile(r"\.stream\(\)"),
    re.compile(
        r"\.(?:filter|map|flatMap|reduce|collect|forEach|peek"
        r"|sorted|distinct|limit|skip|findFirst|findAny"
        r"|anyMatch|allMatch|noneMatch|count|toList)\("
    ),
    re.compile(r"\.collect\(\s*Collectors\."),
    re.compile(r"Optional\.(?:of|ofNullable|empty|isPresent|ifPresent|orElse|orElseThrow|orElseGet)\("),

    # Logging patterns
    re.compile(
        r"(?:logger|log|LOG|LOGGER)\."
        r"(?:trace|debug|info|warn|error|fatal)\("
    ),

    # Builder / factory patterns
    re.compile(r"\.builder\(\)"),
    re.compile(r"\.build\(\)"),

    # Null checking
    re.compile(r"Objects\.requireNonNull\("),

    # Return patterns
    re.compile(
        r"return\s+(?:null|this|true|false|super\.\w+"
        r"|Optional\.empty\(\)|Collections\.(?:emptyList|emptyMap|emptySet)\(\)"
        r"|Collections\.unmodifiable\w+\()\s*;"
    ),

    # Javadoc tags (compressible because Javadoc is safe)
    re.compile(r"@(?:param|return|throws|exception|see|since|version|author|deprecated)\s"),
    re.compile(r"\{@(?:link|code|literal|value|inheritDoc)[\s}]"),

    # Try-with-resources
    re.compile(r"try\s*\("),

    # Test assertion patterns
    re.compile(
        r"(?:assertEquals|assertNotEquals|assertTrue|assertFalse"
        r"|assertNull|assertNotNull|assertThrows|assertThat"
        r"|assertSame|assertArrayEquals|assertIterableEquals)\s*\("
    ),
    re.compile(r"verify\(\s*\w+(?:,\s*\w+\([^)\n]*\))?\s*\)\."),
]

# ---------------------------------------------------------------------------
# Dictionary: seed patterns
# ---------------------------------------------------------------------------

SEED_PATTERNS = [
    # Import patterns (extremely high frequency)
    ("import java.util.List;", "import"),
    ("import java.util.Map;", "import"),
    ("import java.util.Set;", "import"),
    ("import java.util.Optional;", "import"),
    ("import java.util.ArrayList;", "import"),
    ("import java.util.HashMap;", "import"),
    ("import java.util.Collections;", "import"),
    ("import java.util.Objects;", "import"),
    ("import java.util.Arrays;", "import"),
    ("import java.util.stream.Collectors;", "import"),
    ("import java.io.IOException;", "import"),
    ("import java.io.Serializable;", "import"),

    # Annotations (extremely high frequency)
    ("@Override", "annotation"),
    ("@Deprecated", "annotation"),
    ("@SuppressWarnings(\"unchecked\")", "annotation"),
    ("@FunctionalInterface", "annotation"),
    ("@Nullable", "annotation"),
    ("@NonNull", "annotation"),
    ("@Test", "annotation"),
    ("@Bean", "annotation"),
    ("@Autowired", "annotation"),
    ("@Inject", "annotation"),
    ("@Component", "annotation"),
    ("@Service", "annotation"),
    ("@Repository", "annotation"),
    ("@Controller", "annotation"),
    ("@RestController", "annotation"),
    ("@Configuration", "annotation"),
    ("@RequestMapping(", "annotation"),
    ("@Transactional", "annotation"),

    # Access modifier combos
    ("public static final", "modifier"),
    ("private static final", "modifier"),
    ("private final", "modifier"),
    ("public static void main(String[] args)", "modifier"),

    # Exception patterns
    ("throw new IllegalArgumentException(", "exception"),
    ("throw new IllegalStateException(", "exception"),
    ("throw new NullPointerException(", "exception"),
    ("throw new UnsupportedOperationException(", "exception"),
    ("throw new RuntimeException(", "exception"),
    ("throw new IOException(", "exception"),
    ("catch (Exception e)", "exception"),
    ("catch (IOException e)", "exception"),

    # Return patterns
    ("return null;", "expression"),
    ("return this;", "expression"),
    ("return true;", "expression"),
    ("return false;", "expression"),
    ("return Optional.empty();", "expression"),
    ("return Collections.emptyList();", "expression"),
    ("return Collections.emptyMap();", "expression"),
    ("return Collections.unmodifiableList(", "expression"),

    # Stream / functional
    (".stream()", "stream"),
    (".collect(Collectors.", "stream"),
    (".collect(Collectors.toList())", "stream"),
    ("Optional.ofNullable(", "stream"),
    ("Optional.of(", "stream"),

    # Null checks
    ("Objects.requireNonNull(", "expression"),
    ("!= null", "expression"),
    ("== null", "expression"),

    # Javadoc tags
    ("@param ", "javadoc"),
    ("@return ", "javadoc"),
    ("@throws ", "javadoc"),
    ("@see ", "javadoc"),
    ("@since ", "javadoc"),
    ("{@link ", "javadoc"),
    ("{@code ", "javadoc"),
    ("{@inheritDoc}", "javadoc"),

    # Logging
    ("private static final Logger", "logging"),
    ("logger.info(", "logging"),
    ("logger.debug(", "logging"),
    ("logger.warn(", "logging"),
    ("logger.error(", "logging"),
    ("LOG.info(", "logging"),
    ("LOG.debug(", "logging"),
    ("LOG.warn(", "logging"),
    ("LOG.error(", "logging"),

    # Test patterns
    ("assertEquals(", "test"),
    ("assertNotNull(", "test"),
    ("assertTrue(", "test"),
    ("assertFalse(", "test"),
    ("assertThrows(", "test"),
    ("assertThat(", "test"),
]

# ---------------------------------------------------------------------------
# AST mining: subtree root types
# ---------------------------------------------------------------------------

SUBTREE_ROOT_TYPES = {
    # Statements
    "expression_statement",
    "local_variable_declaration",
    "return_statement",
    "throw_statement",
    "if_statement",
    "for_statement",
    "enhanced_for_statement",
    "while_statement",
    "try_statement",
    "try_with_resources_statement",
    "switch_expression",
    "assert_statement",
    # Declarations
    "field_declaration",
    "method_declaration",
    "constructor_declaration",
    # Annotations
    "annotation",
    "marker_annotation",
    # Expressions (as subtree roots for normalization)
    "method_invocation",
    "object_creation_expression",
    "assignment_expression",
    "variable_declarator",
}

# ---------------------------------------------------------------------------
# Template mining: identifier normalization
# ---------------------------------------------------------------------------

# Parent node types where an identifier is structural (defines the pattern)
FIXED_PARENT_TYPES = {
    # Declarations
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "annotation_type_declaration",
    "method_declaration",
    "constructor_declaration",
    # Invocations & access
    "method_invocation",
    "object_creation_expression",
    "field_access",
    "scoped_identifier",
    # Imports & packages
    "import_declaration",
    "package_declaration",
    # Annotations
    "annotation",
    "marker_annotation",
    # Types
    "type_identifier",
    "generic_type",
    "type_arguments",
    # Exception handling
    "catch_clause",
    "throws",
}

# Parent node types where an identifier is a user-chosen name (normalizable)
NORMALIZE_PARENT_TYPES = {
    # Assignments
    "variable_declarator",
    "assignment_expression",
    # Statements
    "return_statement",
    "throw_statement",
    # Arguments
    "argument_list",
    # Expressions
    "binary_expression",
    "unary_expression",
    "ternary_expression",
    "instanceof_expression",
    # Arrays
    "array_access",
    "array_creation_expression",
    # Parenthesized
    "parenthesized_expression",
    # Lambda
    "lambda_expression",
    # Enhanced for loop target
    "enhanced_for_statement",
}

# Well-known names that should never be normalized even if parent says so
STRUCTURAL_NAMES = {
    # Java keywords / literals that tree-sitter may label as identifier
    "this", "super", "null", "true", "false", "void", "class", "new",
    "instanceof", "var",

    # Primitive types
    "int", "long", "short", "byte", "float", "double", "char", "boolean",

    # Boxed / wrapper types
    "Integer", "Long", "Short", "Byte", "Float", "Double", "Character",
    "Boolean", "String", "Object", "Number", "Void",

    # Collection types
    "List", "ArrayList", "LinkedList", "Map", "HashMap", "TreeMap",
    "LinkedHashMap", "ConcurrentHashMap", "Set", "HashSet", "TreeSet",
    "Queue", "Deque", "ArrayDeque", "PriorityQueue", "Stack", "Vector",
    "Collection", "Iterable", "Iterator", "Comparable", "Comparator",
    "Optional", "Stream", "Collectors",

    # Concurrency types
    "Thread", "Runnable", "Callable", "Future", "CompletableFuture",
    "Executor", "ExecutorService", "Lock", "ReentrantLock",
    "AtomicInteger", "AtomicLong", "AtomicReference", "AtomicBoolean",
    "CountDownLatch", "CyclicBarrier", "Semaphore", "BlockingQueue",

    # IO / NIO types
    "File", "Path", "Paths", "Files", "InputStream", "OutputStream",
    "Reader", "Writer", "BufferedReader", "BufferedWriter",
    "FileInputStream", "FileOutputStream", "ByteArrayOutputStream",
    "ByteArrayInputStream", "StringWriter", "StringReader",
    "Closeable", "AutoCloseable", "Serializable",
    "Charset", "StandardCharsets", "URI", "URL",

    # Exception types
    "Exception", "RuntimeException", "IOException",
    "IllegalArgumentException", "IllegalStateException",
    "NullPointerException", "UnsupportedOperationException",
    "IndexOutOfBoundsException", "ClassNotFoundException",
    "ClassCastException", "NoSuchElementException",
    "ConcurrentModificationException", "InterruptedException",
    "TimeoutException", "ExecutionException",
    "Error", "AssertionError", "OutOfMemoryError", "StackOverflowError",

    # Utility classes
    "Objects", "Arrays", "Collections", "Math", "System", "Runtime",
    "Class", "StringBuilder", "StringBuffer", "Pattern", "Matcher",
    "DateTimeFormatter", "LocalDate", "LocalDateTime", "Instant",
    "Duration", "ZonedDateTime", "BigDecimal", "BigInteger", "UUID",

    # Annotation names
    "Override", "Deprecated", "SuppressWarnings", "FunctionalInterface",
    "SafeVarargs", "Nullable", "NonNull", "NotNull",

    # Logging
    "Logger", "LoggerFactory", "LOG", "log", "logger", "LOGGER",

    # Framework / library names
    "Autowired", "Inject", "Bean", "Component", "Service", "Repository",
    "Controller", "RestController", "Configuration", "SpringApplication",
    "Test", "Before", "After", "BeforeEach", "AfterEach",
    "BeforeAll", "AfterAll",

    # Functional interfaces
    "Supplier", "Consumer", "Function", "Predicate", "BiFunction",
    "BiConsumer", "BiPredicate", "UnaryOperator", "BinaryOperator",

    # Common structural method names
    "toString", "hashCode", "equals", "compareTo", "valueOf", "of",
    "get", "set", "put", "add", "remove", "contains", "containsKey",
    "size", "isEmpty", "clear", "iterator", "stream", "close",
    "read", "write", "flush", "append", "format", "parse",
    "build", "builder", "run", "call", "apply", "accept", "test",
    "getName", "setName", "getId", "setId", "getClass",
    "notify", "notifyAll", "wait", "finalize", "clone",
}

# ---------------------------------------------------------------------------
# Download: paths to skip when extracting source files
# ---------------------------------------------------------------------------

SKIP_PATH_PATTERNS = [
    "/target/",
    "/build/",
    ".class",
    "/generated/",
    "/generated-sources/",
    "/generated-test-sources/",
    "module-info.java",
    "package-info.java",
]

# ---------------------------------------------------------------------------
# Data: repos and eval splits
# ---------------------------------------------------------------------------

REPOS = [
    # --- Web frameworks & servers (7) ---
    ("spring-projects", "spring-boot"),           # Apache-2.0
    ("spring-projects", "spring-framework"),       # Apache-2.0
    ("spring-projects", "spring-security"),        # Apache-2.0
    ("quarkusio", "quarkus"),                     # Apache-2.0
    ("micronaut-projects", "micronaut-core"),     # Apache-2.0
    ("dropwizard", "dropwizard"),                 # Apache-2.0
    ("apache", "tomcat"),                         # Apache-2.0
    # --- Enterprise / microservices (5) ---
    ("apache", "dubbo"),                          # Apache-2.0
    ("alibaba", "nacos"),                         # Apache-2.0
    ("resilience4j", "resilience4j"),             # Apache-2.0
    ("OpenFeign", "feign"),                       # Apache-2.0
    ("apolloconfig", "apollo"),                   # Apache-2.0
    # --- Big data / distributed systems (6) ---
    ("apache", "kafka"),                          # Apache-2.0
    ("apache", "flink"),                          # Apache-2.0
    ("apache", "pulsar"),                         # Apache-2.0
    ("apache", "zookeeper"),                      # Apache-2.0
    ("apache", "beam"),                           # Apache-2.0
    ("prestodb", "presto"),                       # Apache-2.0
    # --- Database / ORM / data access (5) ---
    ("mybatis", "mybatis-3"),                     # Apache-2.0
    ("brettwooldridge", "HikariCP"),             # Apache-2.0
    ("alibaba", "druid"),                         # Apache-2.0
    ("apache", "shardingsphere"),                 # Apache-2.0
    ("debezium", "debezium"),                     # Apache-2.0
    # --- Messaging / integration (2) ---
    ("apache", "rocketmq"),                       # Apache-2.0
    ("apache", "camel"),                          # Apache-2.0
    # --- Cloud / observability (3) ---
    ("apache", "skywalking"),                     # Apache-2.0
    ("micrometer-metrics", "micrometer"),         # Apache-2.0
    ("fabric8io", "kubernetes-client"),           # Apache-2.0
    # --- Networking / async / reactive (4) ---
    ("netty", "netty"),                           # Apache-2.0
    ("grpc", "grpc-java"),                        # Apache-2.0
    ("ReactiveX", "RxJava"),                      # Apache-2.0
    ("reactor", "reactor-core"),                  # Apache-2.0
    # --- Android (5) ---
    ("google", "dagger"),                         # Apache-2.0
    ("airbnb", "lottie-android"),                 # Apache-2.0
    ("greenrobot", "EventBus"),                   # Apache-2.0
    ("yuliskov", "SmartTube"),                    # MIT
    ("android10", "Android-CleanArchitecture"),   # Apache-2.0
    # --- Testing (4) ---
    ("mockito", "mockito"),                       # MIT
    ("testcontainers", "testcontainers-java"),   # MIT
    ("assertj", "assertj"),                       # Apache-2.0
    ("wiremock", "wiremock"),                     # Apache-2.0
    # --- Libraries / utilities (7) ---
    ("google", "guava"),                          # Apache-2.0
    ("google", "gson"),                           # Apache-2.0
    ("FasterXML", "jackson-databind"),           # Apache-2.0
    ("ben-manes", "caffeine"),                   # Apache-2.0
    ("jhy", "jsoup"),                             # MIT
    ("redisson", "redisson"),                     # Apache-2.0
    ("google", "guice"),                          # Apache-2.0
    # --- Code generation / annotation processing (3) ---
    ("mapstruct", "mapstruct"),                   # Apache-2.0
    ("google", "auto"),                           # Apache-2.0
    ("square", "javapoet"),                       # Apache-2.0
    # --- Security (2) ---
    ("keycloak", "keycloak"),                     # Apache-2.0
    ("apache", "shiro"),                          # Apache-2.0
    # --- Logging (1) ---
    ("apache", "logging-log4j2"),                 # Apache-2.0
    # --- Build / DevOps / applications (4) ---
    ("jenkinsci", "jenkins"),                     # MIT
    ("apache", "maven"),                          # Apache-2.0
    ("dbeaver", "dbeaver"),                       # Apache-2.0
    ("SeleniumHQ", "selenium"),                   # Apache-2.0
    # --- Game development (2) ---
    ("libgdx", "libgdx"),                         # Apache-2.0
    ("jMonkeyEngine", "jmonkeyengine"),           # BSD-3-Clause
]

EVAL_REPOS = [
    "quarkusio--quarkus",                         # Apache-2.0  | Modern web framework
    "mockito--mockito",                           # MIT         | Testing framework
    "apache--pulsar",                             # Apache-2.0  | Distributed messaging
    "ben-manes--caffeine",                        # Apache-2.0  | Caching library
    "keycloak--keycloak",                         # Apache-2.0  | Identity & access mgmt
    "debezium--debezium",                         # Apache-2.0  | Change data capture
    "resilience4j--resilience4j",                 # Apache-2.0  | Fault tolerance
    "jhy--jsoup",                                 # MIT         | HTML parser / utility
    "libgdx--libgdx",                             # Apache-2.0  | Game development
]
