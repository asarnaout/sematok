"""C# language configuration for sematok."""

import re

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language

from sematok.languages import LanguageConfig


def _is_xmldoc_comment(node, source_bytes: bytes) -> bool:
    """Treat /// XML doc comments as safe (compressible)."""
    if node.type == "comment":
        text = source_bytes[node.start_byte:node.end_byte]
        return text.startswith(b"///")
    return False


def get_config() -> LanguageConfig:
    """Return the C# language configuration."""
    return LanguageConfig(
        name="csharp",
        file_extension=".cs",
        tree_sitter_language=Language(tscsharp.language()),
        unsafe_node_types=UNSAFE_NODE_TYPES,
        is_safe_override=_is_xmldoc_comment,
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
    "comment",
    "string_literal",
    "verbatim_string_literal",
    "interpolated_string_expression",
    "raw_string_literal",
    "character_literal",
}

# ---------------------------------------------------------------------------
# Mining: regex candidate extraction
# ---------------------------------------------------------------------------

CANDIDATE_PATTERNS = [
    # Using directives
    re.compile(r"using\s+[\w.]+;"),

    # Attribute patterns -- require 3+ char name to reject [0], [i], [1] etc.
    # Negative lookbehind: reject if preceded by a word char or ')' to avoid
    # matching array indexers like buffer[length] or func()[result].
    re.compile(r"(?<![)\w])\[\w{3,}(?:\([^)\n]*\))?\]"),

    # Property accessors
    re.compile(r"\{\s*get;\s*(?:(?:private|protected|internal)\s+)?set;\s*\}"),
    re.compile(r"\{\s*get;\s*(?:init|internal\s+set|protected\s+set);\s*\}"),
    re.compile(r"\{\s*get;\s*\}"),

    # Access modifier combos (2-4 keywords before a type/name)
    re.compile(
        r"(?:public|private|protected|internal)\s+"
        r"(?:static\s+)?(?:readonly\s+)?"
        r"(?:virtual\s+|override\s+|abstract\s+|sealed\s+|async\s+)?"
        r"(?:partial\s+)?"
        r"(?:class|struct|interface|enum|record|void|string|int|bool|long|double|float|decimal|byte|char|object"
        r"|Task|Task<\w+>|IActionResult|ActionResult)"
    ),

    # Common method signatures (with parameter lists)
    re.compile(
        r"(?:public|private|protected|internal)\s+"
        r"(?:static\s+)?(?:override\s+)?(?:async\s+)?"
        r"(?:void|string|int|bool|Task|Task<\w+>)\s+"
        r"\w+\([^)\n]{0,80}\)"
    ),

    # Throw patterns
    re.compile(r"throw\s+new\s+\w+(?:Exception|Error)\([^)\n]*\);"),

    # Common framework expressions
    re.compile(r"Console\.(?:Write|WriteLine|ReadLine|Read)\("),
    re.compile(r"return\s+Task\.(?:CompletedTask|FromResult|Delay|Run)\b"),
    re.compile(r"=\s*(?:string\.Empty|new\(\)|default!?|Array\.Empty<\w+>\(\));"),
    re.compile(r"Debug\.(?:Assert|WriteLine|Write)\("),
    re.compile(r"ArgumentNullException\.ThrowIfNull\("),

    # XML doc
    re.compile(
        r"///\s*<(?:summary|/summary|param\s+name=\"[^\"\n]*\"|returns|/returns"
        r"|exception\s+cref=\"[^\"\n]*\"|remarks|/remarks|value|/value"
        r"|inheritdoc\s*/|see\s+cref=\"[^\"\n]*\"\s*/?)>"
    ),

    # Generic type patterns
    re.compile(
        r"(?:IEnumerable|IList|ICollection|IDictionary|IReadOnlyList|IReadOnlyCollection"
        r"|IReadOnlyDictionary|Dictionary|List|HashSet|SortedSet|Queue|Stack"
        r"|ConcurrentDictionary|Task|ValueTask|Func|Action|Lazy"
        r"|ILogger|IOptions|IServiceProvider|IConfiguration"
        r"|ReadOnlySpan|Span|Memory|ReadOnlyMemory"
        r"|Nullable|KeyValuePair)<"
    ),

    # Namespace/class scaffolding
    re.compile(r"namespace\s+[\w.]+"),

    # Common attributes (multi-word, framework-specific)
    re.compile(r"\[(?:MethodImpl|DllImport|MarshalAs|StructLayout|FieldOffset)\([^)\n]+\)\]"),
    re.compile(r"\[(?:Conditional|Obsolete|Description|Category|DefaultValue)\([^)\n]*\)\]"),
    re.compile(
        r"\[(?:Theory|Fact|InlineData|MemberData|ClassData"
        r"|TestMethod|TestClass|TestCategory"
        r"|ConditionalFact|ConditionalTheory"
        r"|ApiController|Route|HttpGet|HttpPost|HttpPut|HttpDelete|HttpPatch"
        r"|Authorize|AllowAnonymous"
        r"|Required|StringLength|Range|MaxLength|MinLength"
        r"|JsonProperty|JsonPropertyName|JsonIgnore"
        r"|Serializable|Flags|Browsable)\b[^]\n]*\]"
    ),

    # Interface implementation declarations
    re.compile(r":\s*(?:IDisposable|IAsyncDisposable|IEquatable<\w+>|IComparable<\w+>|ICloneable|IEnumerable<\w+>)"),

    # Common parameter patterns (boilerplate, not logic)
    re.compile(r"CancellationToken\s+cancellationToken"),
    re.compile(r"IServiceProvider\s+serviceProvider"),
    re.compile(r"ILogger<\w+>\s+logger"),

    # Async/dispose boilerplate
    re.compile(r"ConfigureAwait\(false\)"),
    re.compile(r"\.GetAwaiter\(\)\.GetResult\(\)"),
    re.compile(r"async\s+ValueTask"),
    re.compile(r"async\s+Task<\w+>"),

    # Assertion patterns (test boilerplate)
    re.compile(r"Assert\.(?:Equal|NotEqual|True|False|Null|NotNull|Throws|Contains|Empty|Same|NotSame"
               r"|IsType|IsAssignableFrom|InRange|Collection|Single)\b"),

    # Common pragma/preprocessor
    re.compile(r"#pragma\s+warning\s+(?:disable|restore)\s+[\w,\s]+"),
    re.compile(r"#if\s+!?(?:NET\w*|NETCOREAPP|NETSTANDARD|DEBUG|RELEASE|WINDOWS)"),

    # Null-checking boilerplate
    re.compile(r"\?\?\s*throw\s+new\s+\w+Exception\("),
    re.compile(r"is\s+(?:not\s+)?null"),

    # Generic constraints
    re.compile(r"where\s+\w+\s*:\s*(?:class|struct|notnull|new\(\)|unmanaged)"),
    re.compile(r"where\s+\w+\s*:\s*(?:IComparable|IEquatable|IEnumerable|IDisposable|ICloneable)<\w+>"),

    # String validation methods
    re.compile(r"string\.IsNullOr(?:Empty|WhiteSpace)\("),

    # LINQ terminal methods
    re.compile(r"\.(?:ToList|ToArray|ToDictionary|ToHashSet|FirstOrDefault|SingleOrDefault|LastOrDefault|First|Single|Last|Count|Any|All)\("),

    # Common method calls (zero-arg, pure boilerplate)
    re.compile(r"\.(?:ToString|GetType|GetHashCode|Equals|Dispose|GetAwaiter)\(\)"),
]

# ---------------------------------------------------------------------------
# Dictionary: seed patterns
# ---------------------------------------------------------------------------

SEED_PATTERNS = [
    # Using directives (very high frequency)
    ("using System;", "using"),
    ("using System.Collections.Generic;", "using"),
    ("using System.Linq;", "using"),
    ("using System.Text;", "using"),
    ("using System.Threading.Tasks;", "using"),
    ("using System.IO;", "using"),
    ("using Microsoft.Extensions.DependencyInjection;", "using"),
    ("using Microsoft.AspNetCore.Mvc;", "using"),
    ("using System.Collections;", "using"),
    ("using Xunit;", "using"),

    # Property patterns (extremely high frequency)
    ("{ get; set; }", "property"),
    ("{ get; private set; }", "property"),
    ("{ get; init; }", "property"),
    ("{ get; internal set; }", "property"),
    ("{ get; protected set; }", "property"),

    # Access modifier + keyword combos (high frequency)
    ("public static void", "modifier"),
    ("public static async Task", "modifier"),
    ("public override string ToString()", "modifier"),
    ("public override bool Equals(object", "modifier"),
    ("public override int GetHashCode()", "modifier"),
    ("private readonly", "modifier"),
    ("public abstract class", "modifier"),
    ("public sealed class", "modifier"),
    ("internal static class", "modifier"),
    ("public static class", "modifier"),

    # Common method signatures
    ("public static void Main(string[] args)", "signature"),
    ("static void Main(string[] args)", "signature"),
    ("public void Dispose()", "signature"),
    ("protected virtual void Dispose(bool disposing)", "signature"),

    # Exception patterns
    ("throw new ArgumentNullException(nameof(", "exception"),
    ("throw new NotImplementedException();", "exception"),
    ("throw new InvalidOperationException(", "exception"),
    ("throw new ArgumentException(", "exception"),
    ("throw new NotSupportedException();", "exception"),

    # Common expressions
    ("Console.WriteLine(", "expression"),
    ("Console.ReadLine();", "expression"),
    ("return Task.CompletedTask;", "expression"),
    ("= string.Empty;", "expression"),
    ("= new();", "expression"),
    ("nameof(", "expression"),

    # Attribute patterns
    ("[ApiController]", "attribute"),
    ("[HttpGet]", "attribute"),
    ("[HttpPost]", "attribute"),
    ("[Serializable]", "attribute"),
    ("[Obsolete]", "attribute"),
    ("[TestMethod]", "attribute"),
    ("[Fact]", "attribute"),

    # Generic type patterns
    ("IEnumerable<", "generic"),
    ("IList<", "generic"),
    ("Dictionary<string, ", "generic"),
    ("ILogger<", "generic"),
    ("IOptions<", "generic"),
    ("Task<IActionResult>", "generic"),

    # XML doc patterns
    ("/// <summary>", "xmldoc"),
    ("/// </summary>", "xmldoc"),
    ("/// <param name=\"", "xmldoc"),
    ("/// <returns>", "xmldoc"),
    ("/// <exception cref=\"", "xmldoc"),
]

# ---------------------------------------------------------------------------
# AST mining: subtree root types
# ---------------------------------------------------------------------------

SUBTREE_ROOT_TYPES = {
    # Statements
    "expression_statement",
    "local_declaration_statement",
    "return_statement",
    "throw_statement",
    "throw_expression",
    "if_statement",
    "for_each_statement",
    "using_statement",
    # Declarations
    "field_declaration",
    "property_declaration",
    "constructor_declaration",
    "method_declaration",
    "parameter",
    "attribute_list",
    # Expressions
    "invocation_expression",
    "object_creation_expression",
    "assignment_expression",
}

# ---------------------------------------------------------------------------
# Template mining: identifier normalization
# ---------------------------------------------------------------------------

# Parent node types where an identifier is structural (defines the pattern)
FIXED_PARENT_TYPES = {
    "class_declaration",
    "struct_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "constructor_declaration",
    "method_declaration",
    "property_declaration",
    "object_creation_expression",
    "invocation_expression",
    "variable_declaration",
    "generic_name",
    "using_directive",
    "attribute",
    "base_list",
    "namespace_declaration",
    "qualified_name",
    "type_argument_list",
}

# Parent node types where an identifier is a user-chosen name (normalizable)
NORMALIZE_PARENT_TYPES = {
    "variable_declarator",
    "member_access_expression",
    "assignment_expression",
    "binary_expression",
    "return_statement",
    "argument",
}

# Well-known names that should never be normalized even if parent says so
STRUCTURAL_NAMES = {
    # Keywords tree-sitter may label as identifier
    "nameof", "sizeof", "typeof", "default", "value", "get", "set", "init",
    "add", "remove", "var", "dynamic", "global", "async", "await",
    # Common framework types
    "Console", "Task", "ValueTask", "String", "Object", "Math",
    "List", "Dictionary", "HashSet", "Array", "Tuple",
    "ILogger", "IOptions", "IConfiguration", "IServiceProvider",
    "IEnumerable", "IList", "ICollection", "IDictionary",
    "IDisposable", "IAsyncDisposable", "ICloneable",
    "CancellationToken", "StringBuilder", "EventArgs",
    "Debug", "Assert", "Trace",
    # Common exception types
    "Exception", "ArgumentNullException", "InvalidOperationException",
    "NotImplementedException", "NotSupportedException", "ArgumentException",
    "ArgumentOutOfRangeException", "NullReferenceException",
    "ObjectDisposedException", "OperationCanceledException",
    # Common method names that are structural
    "Dispose", "ToString", "GetHashCode", "Equals", "GetType",
    "ConfigureAwait", "GetAwaiter", "GetResult",
    "ThrowIfNull", "IsNullOrEmpty", "IsNullOrWhiteSpace",
}

# ---------------------------------------------------------------------------
# Download: paths to skip when extracting source files
# ---------------------------------------------------------------------------

SKIP_PATH_PATTERNS = [
    "obj/",
    "bin/",
    ".designer.cs",
    "assemblyinfo.cs",
    "globalassemblyinfo",
]

# ---------------------------------------------------------------------------
# Data: repos and eval splits
# ---------------------------------------------------------------------------

REPOS = [
    # --- Original 24 repos (dotnet / microsoft ecosystem) ---
    ("dotnet", "runtime"),
    ("dotnet", "roslyn"),
    ("dotnet", "aspnetcore"),
    ("dotnet", "efcore"),
    ("dotnet", "maui"),
    ("dotnet", "orleans"),
    ("dotnet", "machinelearning"),
    ("dotnet", "wpf"),
    ("dotnet", "winforms"),
    ("dotnet", "yarp"),
    ("dotnet", "reactive"),
    ("dotnet", "BenchmarkDotNet"),
    ("dotnet", "eShop"),
    ("microsoft", "semantic-kernel"),
    ("microsoft", "garnet"),
    ("JamesNK", "Newtonsoft.Json"),
    ("icsharpcode", "ILSpy"),
    ("ppy", "osu"),
    ("MudBlazor", "MudBlazor"),
    ("Humanizr", "Humanizer"),
    ("autofac", "Autofac"),
    ("nunit", "nunit"),
    ("bchavez", "Bogus"),
    ("spectreconsole", "spectre.console"),
    # --- New repos: web app patterns ---
    ("jasontaylordev", "CleanArchitecture"),
    ("fullstackhero", "dotnet-starter-kit"),
    ("EduardoPires", "EquinoxProject"),
    ("OrchardCMS", "OrchardCore"),          # BSD-3-Clause
    # --- New repos: messaging / event-driven / CQRS ---
    ("BrighterCommand", "Brighter"),
    ("JasperFx", "wolverine"),
    ("JasperFx", "marten"),
    ("dotnetcore", "CAP"),
    ("MassTransit", "MassTransit"),         # Apache-2.0
    # --- New repos: xUnit / testing ---
    ("xunit", "xunit"),                     # Apache-2.0
    ("ThreeMammals", "Ocelot"),
    ("VerifyTests", "Verify"),
    ("domaindrivendev", "Swashbuckle.AspNetCore"),
    # --- New repos: community / diverse conventions ---
    ("kgrzybek", "modular-monolith-with-ddd"),
    ("AngleSharp", "AngleSharp"),
    ("MapsterMapper", "Mapster"),
    ("FluentValidation", "FluentValidation"),  # Apache-2.0
    ("App-vNext", "Polly"),                 # BSD-3-Clause
    ("serilog", "serilog"),                 # Apache-2.0
    ("DapperLib", "Dapper"),                # Apache-2.0
]

EVAL_REPOS = [
    "microsoft--garnet",                       # Cache/storage (Microsoft conventions)
    "kgrzybek--modular-monolith-with-ddd",     # DDD architecture (community, Polish)
    "nunit--nunit",                            # Testing framework (community)
    "JamesNK--Newtonsoft.Json",                # Serialization (community)
    "AngleSharp--AngleSharp",                  # HTML parser (community, German)
    "ThreeMammals--Ocelot",                    # API gateway, middleware (community, xUnit)
    "fullstackhero--dotnet-starter-kit",       # Web app, API + Blazor (community)
]
